import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, NaiveBayes, OneVsRest
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from pyspark.sql.functions import udf, col
from pyspark.sql import functions as F


NUM_FEATURES = 4096

def adjust_vector(vec, min_value):
    if min_value < 0:
        adjusted_values = [x + abs(min_value) for x in vec]
        return Vectors.dense(adjusted_values)
    return vec

adjust_vector_udf = udf(adjust_vector, VectorUDT())

class SparkML:
    def __init__(self, data_path):
        self.spark = SparkSession.builder \
            .appName("Optimized Spark ML with 48 Cores") \
            .master("local[*]") \
            .config("spark.driver.memory", "16g") \
            .config("spark.executor.memory", "50g") \
            .config("spark.executor.instances", "1") \
            .config("spark.executor.cores", "12") \
            .config("spark.memory.fraction", "0.8") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "20g") \
            .config("spark.executor.memoryOverhead", "10g") \
            .config("spark.sql.debug.maxToStringFields", "1000") \
            .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
            .config("spark.driver.maxResultSize", "4g") \
            .config("spark.sql.shuffle.partitions", "100") \
            .config("spark.default.parallelism", "48") \
            .getOrCreate()
        self.data_path = data_path
        self.df = self.load_and_prepare_data()

    def load_and_prepare_data(self):
        df = self.spark.read.parquet(self.data_path)
        if "target" not in df.columns or not isinstance(df.schema["target"].dataType, StringType):
            raise ValueError("A coluna 'target' é necessária e deve ser do tipo string.")
        label_indexer = StringIndexer(inputCol="target", outputCol="label")
        df = label_indexer.fit(df).transform(df)

        # Lista de todas as colunas de características
        feature_columns = [f"feature_{i}" for i in range(NUM_FEATURES)]

        # Configurando o VectorAssembler para combinar as colunas de características
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        df = assembler.transform(df)

        # Escalando os recursos
        scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
        df = scaler.fit(df).transform(df)

        # Removendo colunas desnecessárias e renomeando as colunas escaladas de volta para 'features'
        return df.drop("features").withColumnRenamed("scaledFeatures", "features")


    def configure_models(self):
        logistic_regression = LogisticRegression(featuresCol='features', labelCol='label', family='multinomial')
        lrParamGrid = ParamGridBuilder().addGrid(logistic_regression.regParam, [0.01, 0.1]).build()

        random_forest = RandomForestClassifier(featuresCol='features', labelCol='label')
        rfParamGrid = ParamGridBuilder().addGrid(random_forest.numTrees, [20]).addGrid(random_forest.maxDepth, [5]).build()

        decision_tree = DecisionTreeClassifier(featuresCol='features', labelCol='label')
        dtParamGrid = ParamGridBuilder().addGrid(decision_tree.maxDepth, [5]).build()

        naive_bayes = NaiveBayes(featuresCol='features', labelCol='label', modelType="multinomial")
        nbParamGrid = ParamGridBuilder().addGrid(naive_bayes.smoothing, [1.0]).build()

        one_vs_rest = OneVsRest(classifier=logistic_regression)
        ovrParamGrid = ParamGridBuilder().addGrid(logistic_regression.regParam, [0.01, 0.1]).build()

        return [
            ("Random Forest", random_forest, rfParamGrid),
            ("Logistic Regression", logistic_regression, lrParamGrid),
            ("Decision Tree", decision_tree, dtParamGrid),
            ("Naive Bayes", naive_bayes, nbParamGrid),
            ("One-vs-Rest", one_vs_rest, ovrParamGrid)
        ]

    def train_and_evaluate_models(self):
        train, test = self.df.randomSplit([0.8, 0.2], seed=42)
        model_directory = "saved_models"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        results = []
        for name, model, paramGrid in self.configure_models():
            evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
            crossval = CrossValidator(estimator=model, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
            cvModel = crossval.fit(train)
            test_pred = cvModel.transform(test)
            
            accuracy = evaluator.evaluate(test_pred, {evaluator.metricName: "accuracy"})
            precision = evaluator.evaluate(test_pred, {evaluator.metricName: "weightedPrecision"})
            recall = evaluator.evaluate(test_pred, {evaluator.metricName: "weightedRecall"})
            f1 = evaluator.evaluate(test_pred, {evaluator.metricName: "f1"})
            
            # Define the model path for saving the model and metrics
            model_path = os.path.join(model_directory, f"model_{name.replace(' ', '_').lower()}")
            cvModel.bestModel.save(model_path)
            
            # Save confusion matrix and metrics
            pred_and_labels = test_pred.select("prediction", "label").rdd.map(lambda x: (float(x[0]), float(x[1])))
            metrics = MulticlassMetrics(pred_and_labels)
            confusion_matrix = metrics.confusionMatrix().toArray()
            np.savetxt(f"{model_path}_confusion_matrix.csv", confusion_matrix, delimiter=",")
            
            # Saving the performance metrics to a CSV file
            metrics_data = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                'Value': [accuracy, precision, recall, f1]
            })
            metrics_data.to_csv(f"{model_path}_metrics.csv", index=False)
            
            print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
            results.append((name, accuracy, precision, recall, f1, confusion_matrix))
        return results

    def plot_metrics(self, results):
        fig, ax = plt.subplots(figsize=(10, 6))
        models = [x[0] for x in results]
        accuracies = [x[1] for x in results]
        ax.barh(models, accuracies, color='skyblue')
        ax.set_title('Model Accuracy Comparison')
        ax.set_xlabel('Accuracy')
        plt.tight_layout()
        plt.show()

def main():
    data_path = './train_data.parquet'
    ml_system = SparkML(data_path)
    results = ml_system.train_and_evaluate_models()
    ml_system.plot_metrics(results)

if __name__ == "__main__":
    main()
