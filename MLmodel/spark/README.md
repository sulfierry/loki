## Project Modules Description

### FormatFile Class

The `FormatFile` class provides comprehensive tools for loading, processing, and saving data, particularly focusing on chemical informatics where SMILES notation is converted into numerical fingerprints. This class is built to handle large datasets efficiently using batch processing and parallel computing. It integrates functionalities to read data from CSV files, preprocess SMILES strings into molecular fingerprints, split data into training and testing sets, and save processed data into Parquet format which is optimal for large-scale data processing tasks in Spark.

Key functionalities:
- Load data from CSV files.
- Convert SMILES to molecular fingerprints using RDKit.
- Batch processing of large datasets to optimize memory usage and performance.
- Split the dataset into training and testing sets for model validation.
- Save processed data in Parquet format for efficient handling in distributed computing environments like Apache Spark.


### SparkML Class

The `SparkML` class is designed to facilitate the machine learning model training and evaluation process using Apache Spark. This class includes methods for setting up a Spark session, loading data, preparing the data pipeline with feature transformations, configuring multiple machine learning models for multiclass classification, and evaluating these models. The class uses Spark's MLlib to create a pipeline that includes vector assembly, scaling, PCA transformation, and model training using cross-validation. The class also provides functionalities to save trained models and visualize their performance metrics.

Key functionalities:
- Initialize a Spark session with customized settings.
- Load and preprocess data from a Parquet file.
- Configure and apply a series of transformations (Vector Assembler, Standard Scaler, PCA, MinMaxScaler) to prepare the data for modeling.
- Set up multiple machine learning models suitable for multiclass classification tasks.
- Train and evaluate models using cross-validation and save the best-performing models.
- Plot and compare the accuracy of different models using matplotlib.

These classes are part of a larger framework designed to streamline the workflow from raw data processing to machine learning model training and evaluation, making it well-suited for projects involving large datasets and complex machine learning tasks.
