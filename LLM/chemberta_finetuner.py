import os
import torch
import numpy as np
import pandas as pd
import psutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, latent_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

# Definindo o dispositivo como GPU (CUDA) se disponível, senão será CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, labels):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(smiles, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        tokens = {key: val.squeeze(0) for key, val in tokens.items()}
        return tokens, label

class ChemBERTaFineTuner:
    def __init__(self, data_path, model_name='seyonec/ChemBERTa-zinc-base-v1', batch_size=32, epochs=10, learning_rate=5e-5, latent_dim=128):
        # Detectar os recursos da máquina
        num_cores = psutil.cpu_count(logical=True)
        total_memory = psutil.virtual_memory().total // (1024 ** 3)  # Convertendo para GB

        self.spark = SparkSession.builder \
            .appName("ChemBERTa Fine-Tuning with Spark") \
            .master(f"local[{num_cores}]") \
            .config("spark.driver.memory", f"{int(total_memory * 0.8)}g") \
            .config("spark.executor.memory", f"{int(total_memory * 0.8)}g") \
            .config("spark.executor.instances", f"{num_cores // 12}") \
            .config("spark.executor.cores", f"{num_cores // 8}") \
            .config("spark.memory.fraction", "0.8") \
            .config("spark.executor.memoryOverhead", f"{int(total_memory * 0.1)}g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", f"{int(total_memory * 0.2)}g") \
            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
            .config("spark.sql.debug.maxToStringFields", "200") \
            .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
            .config("spark.driver.maxResultSize", f"{int(total_memory * 0.1)}g") \
            .config("spark.sql.shuffle.partitions", f"{num_cores * 4}") \
            .config("spark.default.parallelism", f"{num_cores * 2}") \
            .getOrCreate()

        self.data_path = data_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.device = device

        # Carregar modelo e tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.autoencoder = Autoencoder(768, latent_dim)  # Tamanho da camada de saída do Roberta é 768
        self.model.to(self.device)
        self.autoencoder.to(self.device)

    def load_data(self):
        # Carregar dados com Spark
        df = self.spark.read.parquet(self.data_path)
        df = df.select(col("canonical_smiles"), col("target"))

        # Converter para Pandas DataFrame para usar com PyTorch DataLoader
        df_pandas = df.toPandas()
        smiles = df_pandas['canonical_smiles'].tolist()
        labels = df_pandas['target'].astype('category').cat.codes.tolist()

        # Dividir em conjuntos de treino e teste
        smiles_train, smiles_test, labels_train, labels_test = train_test_split(smiles, labels, test_size=0.2, random_state=42)

        # Criar datasets e dataloaders
        train_dataset = SMILESDataset(smiles_train, labels_train)
        test_dataset = SMILESDataset(smiles_test, labels_test)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def train_autoencoder(self):
        self.autoencoder.train()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Training Autoencoder Epoch {epoch + 1}/{self.epochs}"):
                tokens, _ = batch
                tokens = {key: val.to(self.device) for key, val in tokens.items()}

                with torch.no_grad():
                    embeddings = self.model(**tokens).last_hidden_state.mean(dim=1)

                outputs = self.autoencoder(embeddings)
                loss = criterion(outputs, embeddings)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1} Autoencoder Loss: {epoch_loss / len(self.train_loader)}")

    def train_classifier(self):
        self.model.eval()
        self.autoencoder.eval()
        classifier = nn.Linear(self.latent_dim, len(set(self.train_loader.dataset.labels)))
        classifier.to(self.device)
        optimizer = optim.Adam(classifier.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            epoch_loss = 0
            classifier.train()
            for batch in tqdm(self.train_loader, desc=f"Training Classifier Epoch {epoch + 1}/{self.epochs}"):
                tokens, labels = batch
                tokens = {key: val.to(self.device) for key, val in tokens.items()}
                labels = labels.to(self.device)

                with torch.no_grad():
                    embeddings = self.model(**tokens).last_hidden_state.mean(dim=1)
                    latent_vectors = self.autoencoder.encode(embeddings)

                outputs = classifier(latent_vectors)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1} Classifier Loss: {epoch_loss / len(self.train_loader)}")

    def evaluate(self):
        self.model.eval()
        self.autoencoder.eval()
        classifier = nn.Linear(self.latent_dim, len(set(self.train_loader.dataset.labels)))
        classifier.to(self.device)
        correct_predictions = 0
        total_predictions = 0

        classifier.eval()
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                tokens, labels = batch
                tokens = {key: val.to(self.device) for key, val in tokens.items()}
                labels = labels.to(self.device)

                embeddings = self.model(**tokens).last_hidden_state.mean(dim=1)
                latent_vectors = self.autoencoder.encode(embeddings)
                outputs = classifier(latent_vectors)
                predictions = torch.argmax(outputs, dim=1)

                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        accuracy = correct_predictions / total_predictions
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.autoencoder.state_dict(), os.path.join(output_dir, "autoencoder.pt"))

def main():
    data_path = './train_data.parquet'  # Caminho para o arquivo Parquet produzido pela classe FormatFileML
    fine_tuner = ChemBERTaFineTuner(data_path)
    fine_tuner.load_data()
    fine_tuner.train_autoencoder()
    fine_tuner.train_classifier()
    fine_tuner.evaluate()
    fine_tuner.save_model('./finetuned_chemberta')

if __name__ == "__main__":
    main()
