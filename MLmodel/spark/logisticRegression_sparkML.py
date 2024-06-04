import os
import psutil
import time
import logging
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from sklearn.metrics import confusion_matrix

NUM_FEATURES = 5120

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def adjust_vector(vec, min_value):
    if min_value < 0:
        adjusted_values = [x + abs(min_value) for x in vec]
        return Vectors.dense(adjusted_values)
    return vec

adjust_vector_udf = udf(adjust_vector, VectorUDT())

class SparkML:
    def __init__(self, data_path):
        # Detectar os recursos da máquina
        num_cores = psutil.cpu_count(logical=True)
        total_memory = psutil.virtual_memory().total // (1024 ** 3)  # Convertendo para GB

        self.spark = SparkSession.builder \
            .appName("Advanced Spark ML with Logistic Regression") \
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

        self.spark.sparkContext.setLogLevel("ERROR")  # Definindo o nível de log para ERROR
        self.data_path = data_path
        self.df = self.load_and_prepare_data()

    def load_and_prepare_data(self):
        df = self.spark.read.parquet(self.data_path)

        if "target" not in df.columns or not isinstance(df.schema["target"].dataType, StringType):
            raise ValueError("A coluna 'target' é necessária e deve ser do tipo string.")

        label_indexer = StringIndexer(inputCol="target", outputCol="label")
        df = label_indexer.fit(df).transform(df)

        feature_columns = [f"feature_{i}" for i in range(NUM_FEATURES)]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="rawFeatures")
        df = assembler.transform(df)

        scaler = MinMaxScaler(inputCol="rawFeatures", outputCol="features")
        df = scaler.fit(df).transform(df)

        df = df.drop("rawFeatures")

        return df

    def configure_model(self):
        logistic_regression = LogisticRegression(featuresCol='features', labelCol='label', family='multinomial')
        paramGrid = (ParamGridBuilder()
                     .addGrid(logistic_regression.regParam, [0.1, 0.01, 0.001])
                     .addGrid(logistic_regression.elasticNetParam, [0.0, 0.1, 0.5])
                     .addGrid(logistic_regression.maxIter, [50, 100])
                     .addGrid(logistic_regression.tol, [1e-4, 1e-5])
                     .addGrid(logistic_regression.fitIntercept, [True, False])
                     .addGrid(logistic_regression.standardization, [True, False])
                     .addGrid(logistic_regression.aggregationDepth, [2, 5])
                     .build())

        return logistic_regression, paramGrid

    def train_and_evaluate_model(self):
        train, test = self.df.randomSplit([0.8, 0.2], seed=42)
        results = []
        model_directory = "saved_models"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        metrics = ['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
        evaluators = {metric: MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName=metric) for metric in metrics}

        paramGrid = self.configure_model()[1]

        with open("result_sparkML.tsv", "w") as f:
            f.write("Model\tParams\tTrain Accuracy\tTest Accuracy\tTrain Weighted Precision\tTest Weighted Precision\tTrain Weighted Recall\tTest Weighted Recall\tTrain F1\tTest F1\n")

            name = "Logistic Regression"
            model, paramGrid = self.configure_model()
            crossval = CrossValidator(estimator=model, estimatorParamMaps=paramGrid, evaluator=evaluators['accuracy'], numFolds=5, parallelism=psutil.cpu_count(logical=True))

            # Medindo o tempo de execução
            start_time = time.time()

            logger.info("Iniciando a validação cruzada...")
            cvModel = crossval.fit(train)

            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Tempo total de execução: {execution_time:.2f} segundos")

            best_model = cvModel.bestModel
            best_params = best_model.extractParamMap()

            # Avaliação no conjunto de treino
            train_pred = cvModel.transform(train)
            train_metrics = {metric: evaluators[metric].evaluate(train_pred) for metric in metrics}

            # Avaliação no conjunto de teste
            test_pred = cvModel.transform(test)
            test_metrics = {metric: evaluators[metric].evaluate(test_pred) for metric in metrics}

            # Calcular e salvar a matriz de confusão
            y_true = [int(row['label']) for row in test.select('label').collect()]
            y_pred = [int(row['prediction']) for row in test_pred.select('prediction').collect()]
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(cm)
            cm_df.to_csv(os.path.join(model_directory, f"confusion_matrix_{name.replace(' ', '_').lower()}.csv"), index=False)

            results.append((name, train_metrics, test_metrics, best_params))
            logger.info(f"{name} - Train Metrics: {train_metrics}")
            logger.info(f"{name} - Test Metrics: {test_metrics}")

            f.write(f"{name}\t{best_params}\t{train_metrics['accuracy']}\t{test_metrics['accuracy']}\t{train_metrics['weightedPrecision']}\t{test_metrics['weightedPrecision']}\t{train_metrics['weightedRecall']}\t{test_metrics['weightedRecall']}\t{train_metrics['f1']}\t{test_metrics['f1']}\n")

            # Plotando os resultados de acurácia em tempo real
            plt.ion()
            fig, ax = plt.subplots()
            x_vals = [str(best_params)]
            y_vals_train = [train_metrics['accuracy']]
            y_vals_test = [test_metrics['accuracy']]

            ax.plot(x_vals, y_vals_train, marker='o', linestyle='-', color='b', label='Train Accuracy')
            ax.plot(x_vals, y_vals_test, marker='x', linestyle='--', color='r', label='Test Accuracy')
            ax.set_xlabel('Hyperparameter Setup')
            ax.set_ylabel('Accuracy')
            ax.set_title('Real-time Hyperparameter Tuning Results')
            ax.legend()
            plt.draw()
            plt.pause(0.1)

        plt.ioff()
        plt.show()

        results.sort(key=lambda x: x[2]['accuracy'], reverse=True)  # Ordenar pela acurácia do teste

        # Salvando resultados
        results_df = pd.DataFrame({
            'Params': [str(r[3]) for r in results],
            'Train Accuracy': [r[1]['accuracy'] for r in results],
            'Test Accuracy': [r[2]['accuracy'] for r in results],
            'Train Weighted Precision': [r[1]['weightedPrecision'] for r in results],
            'Test Weighted Precision': [r[2]['weightedPrecision'] for r in results],
            'Train Weighted Recall': [r[1]['weightedRecall'] for r in results],
            'Test Weighted Recall': [r[2]['weightedRecall'] for r in results],
            'Train F1': [r[1]['f1'] for r in results],
            'Test F1': [r[2]['f1'] for r in results]
        })

        results_df.to_csv('hyperparameter_results.tsv', sep='\t', index=False)

        # Plotando os resultados finais
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(results_df['Params'], results_df['Test Accuracy'], marker='o', linestyle='-', color='b', label='Test Accuracy')
        ax.plot(results_df['Params'], results_df['Train Accuracy'], marker='x', linestyle='--', color='r', label='Train Accuracy')
        ax.set_xlabel('Hyperparameter Setup')
        ax.set_ylabel('Accuracy')
        ax.set_title('Final Hyperparameter Tuning Results')
        plt.xticks(rotation=90)
        ax.legend()
        plt.tight_layout()
        plt.savefig('hyperparameter_tuning_results.png')
        plt.show()

        return results

def main():
    data_path = './train_data.parquet'
    ml_system = SparkML(data_path)
    results = ml_system.train_and_evaluate_model()
    ml_system.plot_metrics(results)

if __name__ == "__main__":
    main(
