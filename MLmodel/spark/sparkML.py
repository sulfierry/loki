import os
import matplotlib.pyplot as plt

# pyspark sql
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, when, isnan, array, count
from pyspark.sql.types import FloatType, BooleanType, ArrayType, StringType

# pyspark ml
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler, PCA, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, NaiveBayes, OneVsRest



# UDF para converter vetor em array e corrigir valores
def vector_to_corrected_array(vector):
    # Corrige o vetor: substitui negativos e NaNs por zero
    return [0 if x < 0 or isnan(x) else x for x in vector.toArray()]

vector_to_corrected_array_udf = udf(vector_to_corrected_array, ArrayType(FloatType()))

def check_and_normalize_vectors(df, feature_col):
    # Converte o vetor de características para array e aplica correções
    df = df.withColumn("corrected_array", vector_to_corrected_array_udf(col(feature_col)))
    
    # Reassemble as características corrigidas de volta para um vetor
    assembler = VectorAssembler(inputCols=["corrected_array"], outputCol="corrected_features")
    df = assembler.transform(df)
    
    # Limpeza das colunas intermediárias
    df = df.drop("corrected_array")

    return df

class SparkML:
    def __init__(self, data_path):
        self.spark = SparkSession.builder \
            .appName("Advanced Spark ML with Multiple Metrics and Models") \
            .master("local[32]") \
            .config("spark.driver.memory", "32g") \
            .config("spark.executor.memory", "32g") \
            .config("spark.executor.instances", "1") \
            .config("spark.executor.cores", "16") \
            .config("spark.memory.fraction", "0.8") \
            .config("spark.executor.memoryOverhead", "4g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "8g") \
            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
            .config("spark.sql.debug.maxToStringFields", "200") \
            .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.default.parallelism", "64") \
            .getOrCreate()

        self.data_path = data_path
        self.df = self.load_and_prepare_data()

    def load_and_prepare_data(self):
        df = self.spark.read.parquet(self.data_path)

        # Verificação adicional: garantir que 'target' existe e é uma string
        if "target" not in df.columns or not isinstance(df.schema["target"].dataType, StringType):
            raise ValueError("A coluna 'target' é necessária e deve ser do tipo string.")

        # Indexando os rótulos que são strings
        label_indexer = StringIndexer(inputCol="target", outputCol="label")
        df = label_indexer.fit(df).transform(df)

        # Assemble features into a single vector
        feature_columns = [f"feature_{i}" for i in range(2048)]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="rawFeatures")
        df = assembler.transform(df)

        # Normalize features to be between 0 and 1
        scaler = MinMaxScaler(inputCol="rawFeatures", outputCol="scaledFeatures")
        df = scaler.fit(df).transform(df)

        # Apply corrections for any improper values
        df = check_and_normalize_vectors(df, "scaledFeatures")

        # Applying PCA
        pca = PCA(k=500, inputCol="scaledFeatures", outputCol="pcaFeatures")  # Using 500 components
        df = pca.fit(df).transform(df)

        # Cleanup intermediate columns not needed for modeling
        df = df.drop("rawFeatures", "scaledFeatures")

        print("Data loaded and prepared with PCA applied! \n")

        return df



    def configure_models(self):
        # Configurando o Logistic Regression
        print("Configuring logistic regression \n")

        logistic_regression = LogisticRegression(featuresCol='features', labelCol='label', family='multinomial')
        lrParamGrid = (ParamGridBuilder()
                       .addGrid(logistic_regression.regParam, [0.0])  # Regularização
                       .addGrid(logistic_regression.elasticNetParam, [0.0])  # ElasticNet mixing
                       .build())

        print("Configuring Random Forest \n")
        # Configurando o Random Forest
        random_forest = RandomForestClassifier(featuresCol='features', labelCol='label')
        rfParamGrid = (ParamGridBuilder()
                       .addGrid(random_forest.numTrees, [20])  # Número de árvores
                       .addGrid(random_forest.maxDepth, [5])  # Profundidade máxima
                       .build())

        print("Configuring decision tree \n")
        # Configurando o Decision Tree
        decision_tree = DecisionTreeClassifier(featuresCol='features', labelCol='label')
        dtParamGrid = (ParamGridBuilder()
                       .addGrid(decision_tree.maxDepth, [5])  # Profundidade máxima
                       .build())

        print("Configuring naive bayes \n")
        # Configurando o Naive Bayes
        naive_bayes = NaiveBayes(featuresCol='features', labelCol='label', modelType="multinomial")
        nbParamGrid = (ParamGridBuilder()
                       .addGrid(naive_bayes.smoothing, [1.0])  # Suavização
                       .build())

        print("Configuring one-vs-rest \n")
        # Configurando o One-vs-Rest
        one_vs_rest = OneVsRest(classifier=logistic_regression)
        ovrParamGrid = (ParamGridBuilder()
                       .addGrid(logistic_regression.regParam, [0.0])  # Pode reutilizar os parâmetros de logistic regression
                       .addGrid(logistic_regression.elasticNetParam, [0.0])
                       .build())

        return [
            ("Random Forest", random_forest, rfParamGrid),
            ("Logistic Regression", logistic_regression, lrParamGrid),
            ("Decision Tree", decision_tree, dtParamGrid),
            ("Naive Bayes", naive_bayes, nbParamGrid),
            ("One-vs-Rest", one_vs_rest, ovrParamGrid)
        ]        


    def train_and_evaluate_models(self):
        print("Train and evaluate methodo start \n")

        train, test = self.df.randomSplit([0.8, 0.2], seed=42)
        results = []
        model_directory = "saved_models"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        for name, model, paramGrid in self.configure_models():
            evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
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

def main():
    data_path = './train_data.parquet'
    ml_system = SparkML(data_path)
    ml_system.train_and_evaluate_models()

if __name__ == "__main__":
    main()
