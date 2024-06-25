import os
import time
import json
import torch
import optuna
import psutil
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup

# Define o número de trabalhadores e o dispositivo de computação (CPU ou GPU)
WORKERS = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Classe personalizada para dataset de SMILES
class SMILESDataset(Dataset):
    def __init__(self, smiles_list, labels, model_name):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(smiles, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        tokens = {key: val.squeeze(0) for key, val in tokens.items()}
        return tokens, label

# Classe para realizar o fine-tuning do modelo ChemBERTa
class ChemBERTaFineTuner:
    def __init__(self, data_path, model_name, batch_size=32, epochs=10, learning_rate=2e-5):
        num_cores = psutil.cpu_count(logical=True)
        total_memory = psutil.virtual_memory().total // (1024 ** 3)  # Convertendo para GB

        # Configuração da sessão Spark
        self.spark = SparkSession.builder \
            .appName("ChemBERTa Fine-Tuning with Spark") \
            .master(f"local[{num_cores}]") \
            .config("spark.driver.memory", f"{int(total_memory * 0.8)}g") \
            .config("spark.executor.memory", f"{int(total_memory * 0.8)}g") \
            .config("spark.executor.instances", f"{num_cores // 4}") \
            .config("spark.executor.cores", f"{num_cores // 4}") \
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
        self.device = DEVICE
        self.scaler = GradScaler()

        # Carregar modelo e tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name, ignore_mismatched_sizes=True)

        # Obter a dimensão correta da saída do modelo Roberta
        self.hidden_size = self.model.config.hidden_size

        # Número de classes (ajustar conforme necessário)
        self.num_classes = self._get_num_classes()

        # Definir classificador com o número correto de classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.model.to(self.device)
        self.classifier.to(self.device)

    # Método para obter o número de classes
    def _get_num_classes(self):
        df = self.spark.read.parquet(self.data_path)
        df = df.select(col("target"))
        df_pandas = df.toPandas()
        return df_pandas['target'].nunique()

    # Método para carregar e preparar os dados
    def load_data(self):
        df = self.spark.read.parquet(self.data_path)
        df = df.select(col("canonical_smiles"), col("target"))

        df_pandas = df.toPandas()
        smiles = df_pandas['canonical_smiles'].tolist()
        labels = df_pandas['target'].astype('category').cat.codes.tolist()

        smiles_train, smiles_test, labels_train, labels_test = train_test_split(smiles, labels, test_size=0.2, random_state=42)

        train_dataset = SMILESDataset(smiles_train, labels_train, self.model_name)
        test_dataset = SMILESDataset(smiles_test, labels_test, self.model_name)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=WORKERS)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=WORKERS)

    # Método para treinamento do modelo
     def train_classifier(self):
        self.model.train()  # Coloca o modelo Roberta em modo de treinamento
        self.classifier.train()  # Coloca o classificador em modo de treinamento
        
        # Define o otimizador AdamW com os parâmetros do modelo e do classificador
        optimizer = optim.AdamW(list(self.model.parameters()) + list(self.classifier.parameters()), lr=self.learning_rate)
        
        # Define o critério de perda como CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        
        # Calcula o número total de passos de treinamento
        total_steps = len(self.train_loader) * self.epochs
        
        # Define o scheduler para ajustar a taxa de aprendizado durante o treinamento
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
        epoch_metrics = []  # Lista para armazenar as métricas de cada época
        
        # Loop de treinamento para cada época
        for epoch in range(self.epochs):
            start_time = time.time()  # Marca o início da época
            epoch_loss = 0  # Inicializa a perda da época
            correct_predictions = 0  # Inicializa a contagem de predições corretas
            total_predictions = 0  # Inicializa a contagem total de predições
            
            all_labels = []  # Lista para armazenar todos os rótulos reais
            all_predictions = []  # Lista para armazenar todas as predições do modelo
            
            # Loop sobre cada mini-batch de dados de treinamento
            for batch in tqdm(self.train_loader, desc=f"Training Classifier Epoch {epoch + 1}/{self.epochs}"):
                tokens, labels = batch  # Extrai os tokens e rótulos do mini-batch
                tokens = {key: val.to(self.device) for key, val in tokens.items()}  # Move os tokens para o dispositivo (CPU/GPU)
                labels = labels.to(self.device)  # Move os rótulos para o dispositivo
                
                optimizer.zero_grad()  # Zera os gradientes do otimizador
                
                with autocast():  # Usa mixed precision para acelerar o treinamento e economizar memória
                    outputs = self.model(**tokens).last_hidden_state.mean(dim=1)  # Passa os tokens pelo modelo Roberta
                    predictions = self.classifier(outputs)  # Passa a saída do Roberta pelo classificador
                    loss = criterion(predictions, labels)  # Calcula a perda entre as predições e os rótulos
                
                self.scaler.scale(loss).backward()  # Calcula os gradientes
                self.scaler.step(optimizer)  # Atualiza os parâmetros do modelo
                self.scaler.update()  # Atualiza o scaler para mixed precision
                scheduler.step()  # Atualiza a taxa de aprendizado
                
                epoch_loss += loss.item()  # Acumula a perda da época
                
                predicted_labels = torch.argmax(predictions, dim=1)  # Obtém as predições do modelo
                all_labels.extend(labels.cpu().numpy())  # Armazena os rótulos reais
                all_predictions.extend(predicted_labels.cpu().numpy())  # Armazena as predições do modelo
                correct_predictions += (predicted_labels == labels).sum().item()  # Conta as predições corretas
                total_predictions += labels.size(0)  # Conta o total de predições
            
            epoch_duration = time.time() - start_time  # Calcula a duração da época
            epoch_loss /= len(self.train_loader)  # Calcula a perda média da época
            accuracy = correct_predictions / total_predictions  # Calcula a acurácia da época
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')  # Calcula precisão, recall e F1-score
            
            # Avaliação no conjunto de teste após cada época
            test_loss, test_accuracy, test_precision, test_recall, test_f1 = self.evaluate_per_epoch()
            
            # Armazena as métricas da época
            epoch_metrics.append({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_accuracy": accuracy,
                "train_precision": precision,
                "train_recall": recall,
                "train_f1": f1,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
                "epoch_duration": epoch_duration
            })
    
            # Impressão das métricas da época
            print(f"Epoch {epoch + 1} Train Loss: {epoch_loss}")
            print(f"Epoch {epoch + 1} Train Accuracy: {accuracy * 100:.2f}%")
            print(f"Epoch {epoch + 1} Train Precision: {precision * 100:.2f}%")
            print(f"Epoch {epoch + 1} Train Recall: {recall * 100:.2f}%")
            print(f"Epoch {epoch + 1} Train F1 Score: {f1 * 100:.2f}%")
            print(f"Epoch {epoch + 1} Test Loss: {test_loss:.4f}")
            print(f"Epoch {epoch + 1} Test Accuracy: {test_accuracy * 100:.2f}%")
            print(f"Epoch {epoch + 1} Test Precision: {test_precision * 100:.2f}%")
            print(f"Epoch {epoch + 1} Test Recall: {test_recall * 100:.2f}%")
            print(f"Epoch {epoch + 1} Test F1 Score: {test_f1 * 100:.2f}%")
            print(f"Epoch {epoch + 1} Duration: {epoch_duration:.2f} seconds")
    
        return epoch_metrics


    # Método para avaliação por época
    def evaluate_per_epoch(self):
        self.model.eval()  # Coloca o modelo em modo de avaliação
        self.classifier.eval()  # Coloca o classificador em modo de avaliação
        correct_predictions = 0  # Inicializa a contagem de predições corretas
        total_predictions = 0  # Inicializa a contagem total de predições
    
        all_labels = []  # Lista para armazenar todos os rótulos reais
        all_predictions = []  # Lista para armazenar todas as predições do modelo
        test_losses = []  # Lista para armazenar as perdas do conjunto de teste
    
        criterion = nn.CrossEntropyLoss()  # Define o critério de perda como CrossEntropyLoss
    
        with torch.no_grad():  # Desativa o cálculo de gradientes, pois estamos em modo de avaliação
            for batch in self.test_loader:  # Itera sobre cada mini-batch do conjunto de teste
                tokens, labels = batch  # Extrai os tokens e rótulos do mini-batch
                tokens = {key: val.to(self.device) for key, val in tokens.items()}  # Move os tokens para o dispositivo (CPU/GPU)
                labels = labels.to(self.device)  # Move os rótulos para o dispositivo
    
                outputs = self.model(**tokens).last_hidden_state.mean(dim=1)  # Passa os tokens pelo modelo Roberta
                predictions = self.classifier(outputs)  # Passa a saída do Roberta pelo classificador
                predicted_labels = torch.argmax(predictions, dim=1)  # Obtém as predições do modelo
    
                loss = criterion(predictions, labels)  # Calcula a perda entre as predições e os rótulos
                test_losses.append(loss.item())  # Armazena a perda do mini-batch
    
                all_labels.extend(labels.cpu().numpy())  # Armazena os rótulos reais
                all_predictions.extend(predicted_labels.cpu().numpy())  # Armazena as predições do modelo
    
                correct_predictions += (predicted_labels == labels).sum().item()  # Conta as predições corretas
                total_predictions += labels.size(0)  # Conta o total de predições
    
        test_loss = np.mean(test_losses)  # Calcula a perda média no conjunto de teste
        test_accuracy = correct_predictions / total_predictions  # Calcula a acurácia no conjunto de teste
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')  # Calcula precisão, recall e F1-score
    
        return test_loss, test_accuracy, test_precision, test_recall, test_f1  # Retorna as métricas calculadas

    # Método para avaliação final do modelo
    def evaluate(self):
        self.model.eval()
        self.classifier.eval()
        correct_predictions = 0
        total_predictions = 0

        all_labels = []
        all_predictions = []
        test_losses = []

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                tokens, labels = batch
                tokens = {key: val.to(self.device) for key, val in tokens.items()}
                labels = labels.to(self.device)

                outputs = self.model(**tokens).last_hidden_state.mean(dim=1)
                predictions = self.classifier(outputs)
                predicted_labels = torch.argmax(predictions, dim=1)

                loss = criterion(predictions, labels)
                test_losses.append(loss.item())

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted_labels.cpu().numpy())

                correct_predictions += (predicted_labels == labels).sum().item()
                total_predictions += labels.size(0)

        accuracy = correct_predictions / total_predictions
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

        test_loss = np.mean(test_losses)

        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Test Precision: {precision * 100:.2f}%")
        print(f"Test Recall: {recall * 100:.2f}%")
        print(f"Test F1 Score: {f1 * 100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")

        return accuracy, precision, recall, f1, test_loss

    # Método para salvar o modelo treinado
    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.classifier.state_dict(), os.path.join(output_dir, "classifier.pt"))

# Função para ler a lista de modelos a partir de um arquivo
def read_models(file_path):
    with open(file_path, 'r') as f:
        models = f.read().splitlines()
    return models

# Função de objetivo para o Optuna
def objective(trial, model_name, data_path):
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 64)
    epochs = trial.suggest_int('epochs', 5, 15)

    fine_tuner = ChemBERTaFineTuner(data_path, model_name=model_name, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)

    fine_tuner.load_data()
    epoch_metrics = fine_tuner.train_classifier()
    accuracy, precision, recall, f1, test_loss = fine_tuner.evaluate()

    return accuracy

# Função principal
def main():
    start_time = time.time()
    models = read_models('pre_trained_models.txt')
    data_path = './train_data.parquet'
    results = {}

    for model in models:
        # Verificar se o modelo já foi treinado
        model_output_dir = f'./finetuned_{model.replace("/", "_")}'
        if os.path.exists(model_output_dir):
            print(f"Model {model} already trained. Skipping...")
            continue

        print(f"Training with model: {model}")
        model_start_time = time.time()
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, model, data_path), n_trials=20)

        best_params = study.best_trial.params
        fine_tuner = ChemBERTaFineTuner(data_path, model_name=model, **best_params)

        fine_tuner.load_data()
        epoch_metrics = fine_tuner.train_classifier()
        accuracy, precision, recall, f1, test_loss = fine_tuner.evaluate()
        fine_tuner.save_model(model_output_dir)

        model_end_time = time.time()
        model_duration = model_end_time - model_start_time
        print(f"Time taken to train model {model}: {model_duration:.2f} seconds")

        metrics = {
            "epoch_metrics": epoch_metrics,
            "test_loss": test_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "best_params": best_params,
            "training_time": model_duration
        }

        results[model] = metrics

        with open(f'metrics_{model.replace("/", "_")}.json', 'w') as f:
            json.dump(metrics, f, indent=4)

    # Compare all models and save the results
    with open('all_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Total time taken to run the script: {total_duration:.2f} seconds")

if __name__ == "__main__":
    main()
