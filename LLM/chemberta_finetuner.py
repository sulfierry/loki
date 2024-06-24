import os
import time
import torch
import numpy as np
import pandas as pd
import psutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import optuna
import json
from sklearn.metrics import precision_recall_fscore_support

WORKERS = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

class ChemBERTaFineTuner:
    def __init__(self, data_path, model_name, batch_size=32, epochs=10, learning_rate=2e-5):
        num_cores = psutil.cpu_count(logical=True)
        total_memory = psutil.virtual_memory().total // (1024 ** 3)  # Convertendo para GB

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

    def _get_num_classes(self):
        df = self.spark.read.parquet(self.data_path)
        df = df.select(col("target"))
        df_pandas = df.toPandas()
        return df_pandas['target'].nunique()

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

    def train_classifier(self):
        self.model.train()
        self.classifier.train()
        optimizer = optim.AdamW(list(self.model.parameters()) + list(self.classifier.parameters()), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        total_steps = len(self.train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        epoch_metrics = []

        for epoch in range(self.epochs):
            start_time = time.time()
            epoch_loss = 0
            correct_predictions = 0
            total_predictions = 0

            all_labels = []
            all_predictions = []

            for batch in tqdm(self.train_loader, desc=f"Training Classifier Epoch {epoch + 1}/{self.epochs}"):
                tokens, labels = batch
                tokens = {key: val.to(self.device) for key, val in tokens.items()}
                labels = labels.to(self.device)

                optimizer.zero_grad()

                with autocast():
                    outputs = self.model(**tokens).last_hidden_state.mean(dim=1)
                    predictions = self.classifier(outputs)
                    loss = criterion(predictions, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()

                epoch_loss += loss.item()

                predicted_labels = torch.argmax(predictions, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted_labels.cpu().numpy())
                correct_predictions += (predicted_labels == labels).sum().item()
                total_predictions += labels.size(0)

            epoch_duration = time.time() - start_time
            epoch_loss /= len(self.train_loader)
            accuracy = correct_predictions / total_predictions
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

            # Avaliação no conjunto de teste após cada epoch
            test_loss, test_accuracy, test_precision, test_recall, test_f1 = self.evaluate_per_epoch()

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

    def evaluate_per_epoch(self):
        self.model.eval()
        self.classifier.eval()
        correct_predictions = 0
        total_predictions = 0

        all_labels = []
        all_predictions = []
        test_losses = []

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in self.test_loader:
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

        test_loss = np.mean(test_losses)
        test_accuracy = correct_predictions / total_predictions
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

        return test_loss, test_accuracy, test_precision, test_recall, test_f1

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

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.classifier.state_dict(), os.path.join(output_dir, "classifier.pt"))

def read_models(file_path):
    with open(file_path, 'r') as f:
        models = f.read().splitlines()
    return models

def objective(trial, model_name, data_path):
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 64)
    epochs = trial.suggest_int('epochs', 5, 15)

    fine_tuner = ChemBERTaFineTuner(data_path, model_name=model_name, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)

    fine_tuner.load_data()
    epoch_metrics = fine_tuner.train_classifier()
    accuracy, precision, recall, f1, test_loss = fine_tuner.evaluate()

    return accuracy

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
