## Project Modules Description

### CheckBalance Class

The `CheckBalance` class is integral to assessing and ensuring the class balance within a dataset, particularly in the context of kinase activity classification. This class is designed to first load the relevant data, prepare labels based on activity thresholds, and subsequently analyze the balance of these classes within the dataset. It features detailed visualization capabilities to help understand the distribution and balance of classes both overall and within specific groups.

Key functionalities:
- Load a dataset from a specified file path, focusing on critical columns relevant to kinase activity analysis.
- Prepare binary labels ('active' or 'inactive') based on a predefined activity threshold, which helps in classifying the compounds.
- Analyze and visualize the overall class balance and the balance by kinase groups using bar charts and histograms, allowing for an intuitive understanding of the data distribution.
- Calculate and report key metrics such as class ratio, entropy of the distribution, and the coefficient of variation to assess the balance and diversity of the dataset.
- Save outputs and filtered datasets for further processing, ensuring that data with specific characteristics are excluded from analysis to maintain dataset integrity.

This class is designed to precede data preprocessing and machine learning tasks in the pipeline, ensuring that the input data for model training in `FormatFile` and `SparkML` classes is well-prepared and balanced. It is especially useful in scenarios where class imbalance could bias the results of machine learning models, providing initial insights and adjustments before deep analysis. 

The `CheckBalance` class not only prepares the data but also sets the stage for robust and fair machine learning model development by addressing potential biases in the dataset upfront.


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

### Workflow Description of `sparkML.py`

The `sparkML.py` script processes data through a series of steps designed to prepare it for machine learning modeling efficiently and effectively. Below is a detailed description of each step and the overall data flow:

#### 1. Data Input
- The script begins by reading raw data from a Parquet file. This data includes various features (`feature_0, feature_1, ..., feature_2047`) and a target column (`target`).

#### 2. Label Indexing
- `StringIndexer` is applied to transform the `target` column from strings to numerical indices, which are necessary for use in machine learning algorithms expecting numerical labels.

#### 3. Feature Assembly
- A `VectorAssembler` combines all individual features (`feature_0` to `feature_2047`) into a single vector column (`rawFeatures`).

#### 4. Normalization
- `MinMaxScaler` is used to scale the values in the `rawFeatures` column, resulting in a new column of normalized features (`scaledFeatures`).

#### 5. Dimensionality Reduction (PCA)
- PCA is applied to the `scaledFeatures` column to reduce the dimensionality of the data, producing a representation in a lower-dimensional latent space (`pcaFeatures`). The number of principal components is set to 500.

#### 6. Final Feature Assembly
- A second `VectorAssembler` is utilized to assemble the `pcaFeatures` into a single final vector column named `features`, which is the final output ready for use in machine learning models.

#### 7. Final Output
- The `features` column contains the final feature vectors prepared for training or predictions in models. Intermediate columns are removed to clean up the DataFrame.

#### Data Workflow Diagram

Here is a visual representation of the data workflow in `sparkML.py`:
    
    +----------------+       +------------------+       +--------------------+
    | Dados Iniciais | ----> | Indexação de     | ----> | Montagem de        |
    | (Parquet File) |       | Rótulos (target) |       | Características    |
    +----------------+       +------------------+       | (rawFeatures)      |
                                                         +--------------------+
                                                                       |
                                                                       v
                                                         +---------------------+
                                                         | Normalização        |
                                                         | (MinMaxScaler)      |
                                                         | -> scaledFeatures   |
                                                         +---------------------+
                                                                       |
                                                                       v
                                                         +---------------------+
                                                         | PCA (Redução de     |
                                                         | Dimensionalidade)   |
                                                         | -> pcaFeatures      |
                                                         +---------------------+
                                                                       |
                                                                       v
                                                         +---------------------+
                                                         | Montagem Final      |
                                                         | (VectorAssembler)   |
                                                         | -> features         |
                                                         +---------------------+
                                                                       |
                                                                       v
                                                         +---------------------+
                                                         | Output Final para   |
                                                         | Modelagem           |
                                                         +---------------------+


These classes are part of a larger framework designed to streamline the workflow from raw data processing to machine learning model training and evaluation, making it well-suited for projects involving large datasets and complex machine learning tasks.

The execution sequence in the overall pipeline is as follows:
1. **CheckBalance**: Ensures data balance and integrity.
2. **FormatFile**: Handles data conversion and preparation for machine learning.
3. **SparkML**: Conducts machine learning model training and evaluation.
