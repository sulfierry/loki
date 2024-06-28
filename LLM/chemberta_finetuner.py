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
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, multilabel_confusion_matrix, roc_auc_score, average_precision_score

# Inputs
MODELS_LIST = 'pre_trained_models.txt'
DATA_PATH = './train_data.parquet'

# Definição dos parâmetros de otimização
LEARNINGRATE_START = 1e-5
LEARNINGRATE_END = 3e-5
EPOCHS_START = 3
EPOCHS_END = 10
BATCH_SIZE = 32
N_TRIALS = 3  # Executa 3 trials para otimização

# Define o número de CPU's e o dispositivo de computação (CPU ou GPU)
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
        return tokens, torch.tensor(label, dtype=torch.float)

# Classe para realizar o fine-tuning do modelo ChemBERTa
class ChemBERTaFineTuner:
    def __init__(self, data_path, model_name, batch_size=32, epochs=5, learning_rate=2e-5, weight_decay=0.01):
        self._configure_spark_session()
        self.data_path = data_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
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

    def _configure_spark_session(self):
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

    # Método para obter o número de classes
    def _get_num_classes(self):
        df = self.spark.read.parquet(self.data_path)
        df = df.select(col("target"))
        df_pandas = df.toPandas()
        return len(set(target for sublist in df_pandas['target'] for target in sublist.split(',')))

    # Método para carregar e preparar os dados
    def load_data(self):
        df = self.spark.read.parquet(self.data_path)
        df = df.select(col("canonical_smiles"), col("target"))

        df_pandas = df.toPandas()
        smiles = df_pandas['canonical_smiles'].tolist()
        targets = df_pandas['target'].apply(lambda x: x.split(',')).tolist()

        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(targets)

        self.class_labels = mlb.classes_

        # Divisão dos dados em treino e teste
        smiles_train, smiles_test, labels_train, labels_test = train_test_split(smiles, labels, test_size=0.2, random_state=42)

        self.smiles_train = smiles_train
        self.labels_train = labels_train
        self.smiles_test = smiles_test
        self.labels_test = labels_test

        test_dataset = SMILESDataset(smiles_test, labels_test, self.model_name)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=WORKERS)

    # Método para treinamento do modelo com validação stratified k-fold
    def train_classifier(self):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        epoch_metrics = []  # Lista para armazenar as métricas de cada época

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.smiles_train, np.argmax(self.labels_train, axis=1))):
            train_smiles = [self.smiles_train[i] for i in train_idx]
            train_labels = [self.labels_train[i] for i in train_idx]
            val_smiles = [self.smiles_train[i] for i in val_idx]
            val_labels = [self.labels_train[i] for i in val_idx]

            train_dataset = SMILESDataset(train_smiles, train_labels, self.model_name)
            val_dataset = SMILESDataset(val_smiles, val_labels, self.model_name)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=WORKERS)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=WORKERS)

            self.model.train()  # Coloca o modelo Roberta em modo de treinamento
            self.classifier.train()  # Coloca o classificador em modo de treinamento

            optimizer = optim.AdamW(list(self.model.parameters()) + list(self.classifier.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay, betas=(0.9, 0.98), eps=1e-6)
            criterion = nn.BCEWithLogitsLoss()
            total_steps = len(train_loader) * self.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

            for epoch in range(self.epochs):
                start_time = time.time()
                epoch_loss = 0
                all_labels = []
                all_predictions = []

                for batch in tqdm(train_loader, desc=f"Training Classifier Fold {fold + 1} Epoch {epoch + 1}/{self.epochs}"):
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
                    predicted_labels = (torch.sigmoid(predictions) > 0.5).float()
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted_labels.cpu().numpy())

                epoch_duration = time.time() - start_time
                epoch_loss /= len(train_loader)
                accuracy = accuracy_score(np.vstack(all_labels), np.vstack(all_predictions))
                precision, recall, f1, _ = precision_recall_fscore_support(np.vstack(all_labels), np.vstack(all_predictions), average='weighted', zero_division=0)

                # Calcula a acurácia para cada classe no conjunto de treino
                train_class_accuracies = self.calculate_class_accuracies(np.vstack(all_labels), np.vstack(all_predictions))
                train_class_avg_accuracy = np.mean(train_class_accuracies)

                # Avaliação no conjunto de validação após cada época
                val_loss, val_accuracy, val_precision, val_recall, val_f1, val_class_accuracies, val_class_avg_accuracy, roc_auc, pr_auc, conf_matrix = self.evaluate_per_epoch(val_loader)

                # Avaliação no conjunto de teste após cada época
                test_loss, test_accuracy, test_precision, test_recall, test_f1, test_class_accuracies, test_class_avg_accuracy, test_roc_auc, test_pr_auc, test_conf_matrix = self.evaluate_per_epoch(self.test_loader)

                metrics = {
                    "fold": fold + 1,
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "train_accuracy": accuracy,
                    "train_precision": precision,
                    "train_recall": recall,
                    "train_f1": f1,
                    "train_class_accuracies": train_class_accuracies,  # Lista de floats
                    "train_class_avg_accuracy": train_class_avg_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "val_class_accuracies": val_class_accuracies,  # Lista de floats
                    "val_class_avg_accuracy": val_class_avg_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "test_f1": test_f1,
                    "test_class_accuracies": test_class_accuracies,  # Lista de floats
                    "test_class_avg_accuracy": test_class_avg_accuracy,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "conf_matrix": conf_matrix,  # Lista de listas
                    "epoch_duration": epoch_duration
                }

                epoch_metrics.append(metrics)

                print(f"Fold {fold + 1} Epoch {epoch + 1} Train Loss: {epoch_loss}")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Train Accuracy: {accuracy * 100:.2f}%")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Train Precision: {precision * 100:.2f}%")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Train Recall: {recall * 100:.2f}%")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Train F1 Score: {f1 * 100:.2f}%")
                for class_idx, class_acc in enumerate(train_class_accuracies):
                    print(f"Fold {fold + 1} Epoch {epoch + 1} Train Class {self.class_labels[class_idx]} Accuracy: {class_acc * 100:.2f}%")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Train Class Average Accuracy: {train_class_avg_accuracy * 100:.2f}%")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Val Loss: {val_loss:.4f}")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Val Accuracy: {val_accuracy * 100:.2f}%")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Val Precision: {val_precision * 100:.2f}%")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Val Recall: {val_recall * 100:.2f}%")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Val F1 Score: {val_f1 * 100:.2f}%")
                for class_idx, class_acc in enumerate(val_class_accuracies):
                    print(f"Fold {fold + 1} Epoch {epoch + 1} Val Class {self.class_labels[class_idx]} Accuracy: {class_acc * 100:.2f}%")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Val Class Average Accuracy: {val_class_avg_accuracy * 100:.2f}%")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Test Loss: {test_loss:.4f}")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Test Accuracy: {test_accuracy * 100:.2f}%")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Test Precision: {test_precision * 100:.2f}%")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Test Recall: {test_recall * 100:.2f}%")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Test F1 Score: {test_f1 * 100:.2f}%")
                for class_idx, class_acc in enumerate(test_class_accuracies):
                    print(f"Fold {fold + 1} Epoch {epoch + 1} Test Class {self.class_labels[class_idx]} Accuracy: {class_acc * 100:.2f}%")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Test Class Average Accuracy: {test_class_avg_accuracy * 100:.2f}%")
                print(f"Fold {fold + 1} Epoch {epoch + 1} Duration: {epoch_duration:.2f} seconds")

        return epoch_metrics

    def calculate_class_accuracies(self, labels, predictions):
        mcm = multilabel_confusion_matrix(labels, predictions)
        with np.errstate(divide='ignore', invalid='ignore'):
            class_accuracies = np.divide(mcm[:, 1, 1], mcm[:, 1, 1] + mcm[:, 0, 1])
            class_accuracies[np.isnan(class_accuracies)] = 0  # Define como 0 onde houver divisão por zero
        return class_accuracies.tolist()  # Convertendo para lista

    # Método para avaliação por época
    def evaluate_per_epoch(self, loader):
        self.model.eval()  # Coloca o modelo em modo de avaliação
        self.classifier.eval()  # Coloca o classificador em modo de avaliação
        all_labels = []  # Lista para armazenar todos os rótulos reais
        all_predictions = []  # Lista para armazenar todas as predições do modelo
        all_probabilities = []  # Lista para armazenar as probabilidades preditas pelo modelo
        test_losses = []  # Lista para armazenar as perdas do conjunto de teste

        criterion = nn.BCEWithLogitsLoss()  # Define o critério de perda como BCEWithLogitsLoss

        with torch.no_grad():  # Desativa o cálculo de gradientes, pois estamos em modo de avaliação
            for batch in loader:  # Itera sobre cada mini-batch do conjunto de teste
                tokens, labels = batch  # Extrai os tokens e rótulos do mini-batch
                tokens = {key: val.to(self.device) for key, val in tokens.items()}  # Move os tokens para o dispositivo (CPU/GPU)
                labels = labels.to(self.device)  # Move os rótulos para o dispositivo

                outputs = self.model(**tokens).last_hidden_state.mean(dim=1)  # Passa os tokens pelo modelo Roberta
                predictions = self.classifier(outputs)  # Passa a saída do Roberta pelo classificador
                predicted_labels = (torch.sigmoid(predictions) > 0.5).float()  # Obtém as predições do modelo
                probabilities = torch.sigmoid(predictions).cpu().numpy()  # Obtém as probabilidades preditas

                loss = criterion(predictions, labels)  # Calcula a perda entre as predições e os rótulos
                test_losses.append(loss.item())  # Armazena a perda do mini-batch

                all_labels.extend(labels.cpu().numpy())  # Armazena os rótulos reais
                all_predictions.extend(predicted_labels.cpu().numpy())  # Armazena as predições do modelo
                all_probabilities.extend(probabilities)  # Armazena as probabilidades preditas

        test_loss = np.mean(test_losses)  # Calcula a perda média no conjunto de teste
        all_labels = np.vstack(all_labels)
        all_predictions = np.vstack(all_predictions)
        all_probabilities = np.vstack(all_probabilities)
        test_accuracy = accuracy_score(all_labels, all_predictions)  # Calcula a acurácia no conjunto de teste
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted', zero_division=0)  # Calcula precisão, recall e F1-score

        # Calcula a acurácia para cada classe
        class_accuracies = self.calculate_class_accuracies(all_labels, all_predictions)
        class_avg_accuracy = np.mean(class_accuracies)

        # Calcula ROC-AUC e PR-AUC
        roc_auc = roc_auc_score(all_labels, all_probabilities, average='weighted', multi_class='ovr')
        pr_auc = average_precision_score(all_labels, all_probabilities, average='weighted')

        # Calcula a matriz de confusão
        conf_matrix = multilabel_confusion_matrix(all_labels, all_predictions)

        return test_loss, test_accuracy, test_precision, test_recall, test_f1, class_accuracies, class_avg_accuracy, roc_auc, pr_auc, conf_matrix.tolist()  # Convertendo para lista

    # Método para avaliação final do modelo
    def evaluate(self):
        # Avaliação final é realizada no conjunto de teste (20% dos dados reservados para teste)
        return self.evaluate_per_epoch(self.test_loader)

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
    learning_rate = trial.suggest_float('learning_rate', LEARNINGRATE_START, LEARNINGRATE_END, log=True)
    epochs = trial.suggest_int('epochs', EPOCHS_START, EPOCHS_END)
    
    # Padronizando batch_size para 32
    batch_size = BATCH_SIZE

    fine_tuner = ChemBERTaFineTuner(data_path, model_name=model_name, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)

    fine_tuner.load_data()
    epoch_metrics = fine_tuner.train_classifier()
    _, accuracy, _, _, _, _, _, _, _, _ = fine_tuner.evaluate()

    return accuracy

# Função principal
def main():
    start_time = time.time()  # Marca o início do tempo de execução do script
    models = read_models(MODELS_LIST)  # Lê a lista de modelos pré-treinados a partir de um arquivo
    data_path = DATA_PATH
    results = {}  # Dicionário para armazenar os resultados de cada modelo

    for model in models:
        # Verifica se o modelo já foi treinado anteriormente
        model_output_dir = f'./finetuned_{model.replace("/", "_")}'
        if os.path.exists(model_output_dir):
            print(f"Model {model} already trained. Skipping...")  # Se o modelo já foi treinado, pula para o próximo
            continue

        print(f"Training with model: {model}")  # Informa qual modelo está sendo treinado
        model_start_time = time.time()  # Marca o início do tempo de treinamento para este modelo

        # Cria um estudo Optuna para otimização dos hiperparâmetros, buscando maximizar a acurácia
        study = optuna.create_study(direction='maximize', study_name=model)
        study.optimize(lambda trial: objective(trial, model, data_path), n_trials=N_TRIALS)  # Executa 3 trials para otimização

        best_trial = study.best_trial  # Obtém o melhor trial encontrado pelo Optuna
        best_params = best_trial.params  # Obtém os melhores hiperparâmetros encontrados pelo Optuna

        # Mensagem de conclusão do melhor trial
        print(f"Best trial {best_trial.number} for model {model} finished with average accuracy: {best_trial.value:.4f} and parameters: {best_params}")

        fine_tuner = ChemBERTaFineTuner(data_path, model_name=model, batch_size=32, **best_params)  # Inicializa o fine-tuner com os melhores parâmetros, batch_size fixo em 32

        fine_tuner.load_data()  # Carrega os dados de treinamento e teste
        epoch_metrics = fine_tuner.train_classifier()  # Treina o classificador e obtém as métricas por época
        accuracy, precision, recall, f1, test_loss, class_accuracies, class_avg_accuracy, roc_auc, pr_auc, conf_matrix = fine_tuner.evaluate()  # Avalia o desempenho do modelo no conjunto de teste
        fine_tuner.save_model(model_output_dir)  # Salva o modelo treinado e o tokenizador

        model_end_time = time.time()  # Marca o fim do tempo de treinamento para este modelo
        model_duration = model_end_time - model_start_time  # Calcula a duração do treinamento
        print(f"Time taken to train model {model}: {model_duration:.2f} seconds")  # Informa o tempo de treinamento

        # Armazena as métricas e informações do treinamento no dicionário de resultados
        metrics = {
            "epoch_metrics": epoch_metrics,
            "test_loss": test_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "class_accuracies": class_accuracies,  # Lista de floats
            "class_avg_accuracy": class_avg_accuracy,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "conf_matrix": conf_matrix,  # Já é uma lista
            "best_params": best_params,
            "training_time": model_duration
        }
        results[model] = metrics

        # Salva as métricas individuais do modelo em um arquivo JSON
        with open(f'metrics_{model.replace("/", "_")}.json', 'w') as f:
            json.dump(metrics, f, indent=4)

    # Gera e salva um relatório consolidado com os resultados de todos os modelos
    with open('all_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    end_time = time.time()  # Marca o fim do tempo de execução do script
    total_duration = end_time - start_time  # Calcula a duração total do script
    print(f"Total time taken to run the script: {total_duration:.2f} seconds")  # Informa o tempo total de execução


if __name__ == "__main__":
    main()
