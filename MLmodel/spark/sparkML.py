import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, NaiveBayes, OneVsRest
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import matplotlib.pyplot as plt
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.functions import col, when


class SparkML:
    def __init__(self, data_path):
        self.spark = SparkSession.builder \
            .appName("Advanced Spark ML with Multiple Metrics and Models") \
            .config("spark.executor.memory", "16g") \
            .config("spark.driver.memory", "8g")  \
            .config("spark.memory.fraction", "0.75") \
            .config("spark.executor.memoryOverhead", "2g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "4g") \
            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
            .config("spark.sql.debug.maxToStringFields", "100") \
            .config("spark.sql.autoBroadcastJoinThreshold", "5242880") \
            .getOrCreate()

        self.data_path = data_path
        self.df = self.load_and_prepare_data()


    def load_and_prepare_data(self):
        df = self.spark.read.parquet(self.data_path)
        df = df.withColumnRenamed("target", "label")

        debug_df(df, "Initial data check")

        # Assemble features into a single vector
        assembler = VectorAssembler(inputCols=[f"feature_{i}" for i in range(2048)], outputCol="rawFeatures")
        df = assembler.transform(df)

        debug_df(df, "Post assembly")

        # Normalize features to be between 0 and 1
        scaler = MinMaxScaler(inputCol="rawFeatures", outputCol="features")
        scalerModel = scaler.fit(df)
        df = scalerModel.transform(df)

        debug_df(df, "Post scaling")

        # Apply PCA
        pca = PCA(k=50, inputCol="features", outputCol="pcaFeatures")
        pcaModel = pca.fit(df)  # Fit PCA model
        df = pcaModel.transform(df)

        debug_df(df, "Post PCA")

        # Convert label into numeric indices
        indexer = StringIndexer(inputCol="label", outputCol="indexedLabel", handleInvalid="keep")
        df = indexer.fit(df).transform(df)

        debug_df(df, "Final dataset before model training")

        return df



    def configure_models(self):
        logistic_regression = LogisticRegression(featuresCol='pcaFeatures', labelCol='indexedLabel', family='multinomial')
        rf = RandomForestClassifier(featuresCol='pcaFeatures', labelCol='indexedLabel')
        dt = DecisionTreeClassifier(featuresCol='pcaFeatures', labelCol='indexedLabel')
        nb = NaiveBayes(featuresCol='pcaFeatures', labelCol='indexedLabel', modelType="multinomial")
        ovr = OneVsRest(classifier=logistic_regression)

        return [
            ("Random Forest", rf, ParamGridBuilder().addGrid(rf.numTrees, [10, 20]).build()),
            ("Logistic Regression", logistic_regression, ParamGridBuilder().addGrid(logistic_regression.regParam, [0.1, 0.01]).build()),
            ("Decision Tree", dt, ParamGridBuilder().build()),
            ("Naive Bayes", nb, ParamGridBuilder().build()),
            ("One-vs-Rest", ovr, ParamGridBuilder().build())
        ]

    def train_and_evaluate_models(self):
        train, test = self.df.randomSplit([0.8, 0.2], seed=42)
        results = []
        model_directory = "saved_models"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        for name, model, paramGrid in self.configure_models():
            evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
            crossval = CrossValidator(estimator=model, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
            cvModel = crossval.fit(train)
            test_pred = cvModel.transform(test)
            accuracy = evaluator.evaluate(test_pred)
            results.append((name, accuracy, cvModel))
            print(f"{name} - Accuracy: {accuracy:.4f}")

        results.sort(key=lambda x: x[1], reverse=True)
        top_models = results[:3]
        for i, (name, accuracy, model) in enumerate(top_models):
            model_path = os.path.join(model_directory, f"model_{name.replace(' ', '_').lower()}")
            model.bestModel.save(model_path)
            print(f"Top {i+1} Model: {name} with accuracy: {accuracy} saved to {model_path}")
        self.plot_metrics(results)

    def plot_metrics(self, results):
        fig, ax = plt.subplots(figsize=(10, 6))
        models = [result[0] for result in results]
        accuracies = [result[1] for result in results]
        ax.barh(models, accuracies, color='skyblue')
        ax.set_title('Model Accuracy Comparison')
        ax.set_xlabel('Accuracy')
        ax.set_xlim(0, 1.0)
        plt.tight_layout()
        plt.show()

def debug_df(df, message=""):
    print(message)
    df.describe().show()
    df.select([count(when(col(c).isNull() | isnan(c) | (col(c) < 0), c)).alias(c) for c in df.columns]).show()



def main():
    data_path = './train_data.parquet'
    ml_system = SparkML(data_path)
    ml_system.train_and_evaluate_models()

if __name__ == "__main__":
    main()
