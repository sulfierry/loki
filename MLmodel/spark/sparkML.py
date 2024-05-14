"""
    Este código define uma classe chamada SparkML que configura, treina e avalia vários modelos de machine learning usando o PySpark. 
    A classe utiliza um conjunto de dados de entrada, realiza a preparação dos dados, configura diferentes modelos de classificação 
    (Regressão Logística, Random Forest, Árvore de Decisão, Naive Bayes e One-vs-Rest), realiza a validação cruzada para ajustar 
    hiperparâmetros, avalia os modelos usando várias métricas (precisão, precisão ponderada, recall ponderado e F1 score), salva as 
    melhores configurações de modelo e gera gráficos comparando as métricas de desempenho dos modelos.
"""

import os
import psutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, NaiveBayes, OneVsRest
from sklearn.metrics import confusion_matrix

NUM_FEATURES = 16

# Função para ajustar o vetor para garantir que todos os valores sejam positivos
def adjust_vector(vec, min_value):
    if min_value < 0:
        adjusted_values = [x + abs(min_value) for x in vec]
        return Vectors.dense(adjusted_values)
    return vec

# Definir uma função UDF (User Defined Function) para o ajuste de vetores
adjust_vector_udf = udf(adjust_vector, VectorUDT())

class SparkML:
    def __init__(self, data_path):
        # Detectar os recursos da máquina
        num_cores = psutil.cpu_count(logical=True)
        total_memory = psutil.virtual_memory().total // (1024 ** 3)  # Convertendo para GB

        # Configurar a sessão Spark com parâmetros específicos para desempenho
        self.spark = SparkSession.builder \
            .appName("Advanced Spark ML with Multiple Metrics and Models") \
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
        self.df = self.load_and_prepare_data()

    # Função para carregar e preparar os dados
    def load_and_prepare_data(self):
        df = self.spark.read.parquet(self.data_path)

        # Verificar se a coluna 'target' existe e é do tipo string
        if "target" not in df.columns or not isinstance(df.schema["target"].dataType, StringType):
            raise ValueError("A coluna 'target' é necessária e deve ser do tipo string.")

        # Indexar a coluna de rótulo
        label_indexer = StringIndexer(inputCol="target", outputCol="label")
        df = label_indexer.fit(df).transform(df)

        # Criar uma lista de nomes de colunas de características
        feature_columns = [f"feature_{i}" for i in range(NUM_FEATURES)]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="rawFeatures")
        df = assembler.transform(df)

        # Escalar os valores das características para ficarem entre 0 e 1
        scaler = MinMaxScaler(inputCol="rawFeatures", outputCol="features")
        df = scaler.fit(df).transform(df)

        # Remover a coluna de características não escaladas
        df = df.drop("rawFeatures")

        return df

    # Função para configurar os modelos de machine learning e seus parâmetros de grade
    def configure_models(self):
        logistic_regression = LogisticRegression(featuresCol='features', labelCol='label', family='multinomial')
        lrParamGrid = (ParamGridBuilder()
                       .addGrid(logistic_regression.regParam, [0.1, 0.01])
                       .addGrid(logistic_regression.elasticNetParam, [0.1, 0.01])
                       .build())

        random_forest = RandomForestClassifier(featuresCol='features', labelCol='label')
        rfParamGrid = (ParamGridBuilder()
                       .addGrid(random_forest.numTrees, [20])
                       .addGrid(random_forest.maxDepth, [5])
                       .build())

        decision_tree = DecisionTreeClassifier(featuresCol='features', labelCol='label')
        dtParamGrid = (ParamGridBuilder()
                       .addGrid(decision_tree.maxDepth, [5])
                       .build())

        naive_bayes = NaiveBayes(featuresCol='features', labelCol='label', modelType="multinomial")
        nbParamGrid = (ParamGridBuilder()
                       .addGrid(naive_bayes.smoothing, [1.0])
                       .build())

        one_vs_rest = OneVsRest(classifier=logistic_regression)
        ovrParamGrid = (ParamGridBuilder()
                       .addGrid(logistic_regression.regParam, [0.1, 0.01])
                       .addGrid(logistic_regression.elasticNetParam, [0.1, 0.01])
                       .build())

        return [
            ("Random Forest", random_forest, rfParamGrid),
            ("Logistic Regression", logistic_regression, lrParamGrid),
            ("Decision Tree", decision_tree, dtParamGrid),
            ("Naive Bayes", naive_bayes, nbParamGrid),
            ("One-vs-Rest", one_vs_rest, ovrParamGrid)
        ]

    # Função para treinar e avaliar os modelos
    def train_and_evaluate_models(self):
        # Dividir os dados em conjuntos de treinamento e teste
        train, test = self.df.randomSplit([0.8, 0.2], seed=42)
        results = []
        model_directory = "saved_models"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        # Definir as métricas de avaliação
        metrics = ['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
        evaluators = {metric: MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName=metric) for metric in metrics}

        for name, model, paramGrid in self.configure_models():
            crossval = CrossValidator(estimator=model, estimatorParamMaps=paramGrid, evaluator=evaluators['accuracy'], numFolds=5)
            cvModel = crossval.fit(train)
            test_pred = cvModel.transform(test)
            model_metrics = {metric: evaluators[metric].evaluate(test_pred) for metric in metrics}

            # Calcular e salvar a matriz de confusão
            y_true = [int(row['label']) for row in test.select('label').collect()]
            y_pred = [int(row['prediction']) for row in test_pred.select('prediction').collect()]
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(cm)
            cm_df.to_csv(os.path.join(model_directory, f"confusion_matrix_{name.replace(' ', '_').lower()}.csv"), index=False)

            results.append((name, model_metrics, cvModel))
            print(f"{name} - Metrics: {model_metrics}")

        # Ordenar os resultados pela precisão e salvar os três melhores modelos
        results.sort(key=lambda x: x[1]['accuracy'], reverse=True)
        top_models = results[:3]
        for i, (name, model_metrics, model) in enumerate(top_models):
            model_path = os.path.join(model_directory, f"model_{name.replace(' ', '_').lower()}")
            model.bestModel.save(model_path)
            print(f"Top {i+1} Model: {name} with accuracy: {model_metrics['accuracy']} saved to {model_path}")

        return results

    # Função para plotar e salvar os gráficos das métricas de avaliação
    def plot_metrics(self, results):
        metrics = ['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
        metric_colors = {
            'accuracy': 'blue',
            'weightedPrecision': 'green',
            'weightedRecall': 'orange',
            'f1': 'red'
        }

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            models = [result[0] for result in results]
            metric_values = [result[1][metric] for result in results]
            ax.barh(models, metric_values, color=metric_colors[metric])
            ax.set_title(f'Model {metric.capitalize()} Comparison')
            ax.set_xlabel(metric.capitalize())
            ax.set_xlim(0, 1.0)
            plt.tight_layout()
            plt.savefig(f"metric_{metric}.png")
            plt.show()

        # Plotar todas as métricas em um único gráfico
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            models = [result[0] for result in results]
            metric_values = [result[1][metric] for result in results]
            ax.bar(models, metric_values, color=metric_colors[metric])
            ax.set_title(f'Model {metric.capitalize()} Comparison')
            ax.set_xlabel('Models')
            ax.set_ylabel(metric.capitalize())
            ax.set_ylim(0, 1.0)
            ax.set_xticklabels(models, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("all_metrics.png")
        plt.show()

def main():
    data_path = './train_data.parquet'
    ml_system = SparkML(data_path)
    results = ml_system.train_and_evaluate_models()
    ml_system.plot_metrics(results)

if __name__ == "__main__":
    main()
