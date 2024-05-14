import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

# pyspark sql
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, BooleanType, ArrayType, StringType
from pyspark.sql.functions import udf, col, when, isnan, array, count, min as _min, max as _max, lit

# pyspark ml
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler, PCA, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, NaiveBayes, OneVsRest

NUMBER_FEATURES = 16

def adjust_vector(vec, min_value):
    # Adiciona o valor absoluto do mínimo a cada elemento do vetor, se o mínimo for negativo
    if min_value < 0:
        adjusted_values = [x + abs(min_value) for x in vec]
        return Vectors.dense(adjusted_values)
    return vec

# Registra a UDF
adjust_vector_udf = udf(adjust_vector, VectorUDT())

class SparkML:
    def __init__(self, data_path):
        self.spark = self.initialize_spark()
        self.data_path = data_path
        self.df = self.load_and_prepare_data()

    def initialize_spark(self):
        import multiprocessing

        num_cores = min(multiprocessing.cpu_count(), 32)
        memory_size_gb = min(int(os.popen('free -g').readlines()[1].split()[1]), 16)
        driver_memory = f"{memory_size_gb}g"
        executor_memory = driver_memory
        executor_cores = min(num_cores // 2, 16)
        num_executors = 1

        return SparkSession.builder \
            .appName("Advanced Spark ML with Multiple Metrics and Models") \
            .master(f"local[{num_cores}]") \
            .config("spark.driver.memory", driver_memory) \
            .config("spark.executor.memory", executor_memory) \
            .config("spark.executor.instances", num_executors) \
            .config("spark.executor.cores", executor_cores) \
            .config("spark.memory.fraction", "0.8") \
            .config("spark.executor.memoryOverhead", "4g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "8g") \
            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
            .config("spark.sql.debug.maxToStringFields", "200") \
            .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.sql.shuffle.partitions", str(num_cores * 4)) \
            .config("spark.default.parallelism", str(num_cores * 2)) \
            .getOrCreate()

    def load_and_prepare_data(self):
        df = self.spark.read.parquet(self.data_path)

        if "target" not in df.columns or not isinstance(df.schema["target"].dataType, StringType):
            raise ValueError("A coluna 'target' é necessária e deve ser do tipo string.")

        label_indexer = StringIndexer(inputCol="target", outputCol="label")
        df = label_indexer.fit(df).transform(df)

        feature_columns = [f"feature_{i}" for i in range(NUMBER_FEATURES)]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="rawFeatures")
        df = assembler.transform(df)

        scaler = MinMaxScaler(inputCol="rawFeatures", outputCol="scaledFeatures")
        df = scaler.fit(df).transform(df)

        pca = PCA(k=NUMBER_FEATURES, inputCol="scaledFeatures", outputCol="pcaFeatures")
        df = pca.fit(df).transform(df)

        scaler_after_pca = MinMaxScaler(inputCol="pcaFeatures", outputCol="features")
        df = scaler_after_pca.fit(df).transform(df)

        df = df.drop("rawFeatures", "scaledFeatures", "pcaFeatures")

        return df

    def configure_models(self):
        print("Configuring logistic regression \n")

        logistic_regression = LogisticRegression(featuresCol='features', labelCol='label', family='multinomial')
        lrParamGrid = (ParamGridBuilder()
                       .addGrid(logistic_regression.regParam, [0.0])
                       .addGrid(logistic_regression.elasticNetParam, [0.0])
                       .build())

        print("Configuring one-vs-rest \n")
        one_vs_rest = OneVsRest(classifier=logistic_regression)
        ovrParamGrid = (ParamGridBuilder()
                       .addGrid(logistic_regression.regParam, [0.0])
                       .addGrid(logistic_regression.elasticNetParam, [0.0])
                       .build())

        print("Configuring Random Forest \n")
        random_forest = RandomForestClassifier(featuresCol='features', labelCol='label')
        rfParamGrid = (ParamGridBuilder()
                       .addGrid(random_forest.numTrees, [20])
                       .addGrid(random_forest.maxDepth, [5])
                       .build())

        print("Configuring decision tree \n")
        decision_tree = DecisionTreeClassifier(featuresCol='features', labelCol='label')
        dtParamGrid = (ParamGridBuilder()
                       .addGrid(decision_tree.maxDepth, [5])
                       .build())

        print("Configuring naive bayes \n")
        naive_bayes = NaiveBayes(featuresCol='features', labelCol='label', modelType="multinomial")
        nbParamGrid = (ParamGridBuilder()
                       .addGrid(naive_bayes.smoothing, [1.0])
                       .build())

        return [
            ("Random Forest", random_forest, rfParamGrid),
            ("Logistic Regression", logistic_regression, lrParamGrid),
            ("Decision Tree", decision_tree, dtParamGrid),
            ("Naive Bayes", naive_bayes, nbParamGrid),
            ("One-vs-Rest", one_vs_rest, ovrParamGrid)
        ]

    def train_and_evaluate_models(self):
        print("Train and evaluate method start \n")

        train, test = self.df.randomSplit([0.8, 0.2], seed=42)
        results = []
        model_directory = "saved_models"
        if os.path.exists(model_directory):
            shutil.rmtree(model_directory)
        os.makedirs(model_directory)

        for name, model, paramGrid in self.configure_models():
            evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
            evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
            evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
            evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

            crossval = CrossValidator(estimator=model, estimatorParamMaps=paramGrid, evaluator=evaluator_accuracy, numFolds=5)
            cvModel = crossval.fit(train)
            test_pred = cvModel.transform(test)

            accuracy = evaluator_accuracy.evaluate(test_pred)
            precision = evaluator_precision.evaluate(test_pred)
            recall = evaluator_recall.evaluate(test_pred)
            f1_score = evaluator_f1.evaluate(test_pred)

            results.append((name, accuracy, precision, recall, f1_score, cvModel))
            print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

        results.sort(key=lambda x: x[1], reverse=True)
        top_models = results[:3]
        for i, (name, accuracy, precision, recall, f1_score, model) in enumerate(top_models):
            model_path = os.path.join(model_directory, f"model_{name.replace(' ', '_').lower()}")
            model.bestModel.save(model_path)
            print(f"Top {i+1} Model: {name} with accuracy: {accuracy} saved to {model_path}")

        self.plot_metrics(results)

    def plot_metrics(self, results):
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
        models = [result[0] for result in results]
        accuracy = [result[1] for result in results]
        precision = [result[2] for result in results]
        recall = [result[3] for result in results]
        f1_score = [result[4] for result in results]

        for name, _, _, _, _, _ in results:
            model_path = os.path.join("saved_models", f"model_{name.replace(' ', '_').lower()}")
            if not os.path.exists(model_path):
                os.makedirs(model_path)

        for metric, data, color in zip(metrics, [accuracy, precision, recall, f1_score], colors):
            for name, value in zip(models, data):
                plt.figure(figsize=(10, 6))
                plt.barh([name], [value], color=color)
                plt.title(f'{metric} Comparison for {name}')
                plt.xlabel(metric)
                plt.xlim(0, 1.0)
                plt.tight_layout()
                model_path = os.path.join("saved_models", f"model_{name.replace(' ', '_').lower()}")
                plt.savefig(os.path.join(model_path, f"{metric.lower()}_comparison.png"))
                plt.close()

        metrics_data = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        }

        for metric, data, color in zip(metrics, [accuracy, precision, recall, f1_score], colors):
            plt.figure(figsize=(10, 6))
            plt.barh(models, data, color=color)
            plt.title(f'{metric} Comparison')
            plt.xlabel(metric)
            plt.xlim(0, 1.0)
            plt.tight_layout()
            plt.savefig(os.path.join("saved_models", f"{metric.lower()}_comparison_all.png"))
            plt.close()

        for metric in metrics:
            plt.figure(figsize=(10, 6))
            for model, metric_data in metrics_data.items():
                plt.plot(models, metrics_data[metric], label=model)
            plt.title(f'{metric} for All Models')
            plt.xlabel('Model')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join("saved_models", f"{metric.lower()}_for_all_models.png"))
            plt.close()


def main():
    data_path = './train_data.parquet'
    ml_system = SparkML(data_path)
    ml_system.train_and_evaluate_models()

if __name__ == "__main__":
    main()
