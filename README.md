# Loki (Looking On Kinase Inhibitors)

**Loki** is not just a toolkit, but a comprehensive server designed to provide valuable information on kinase inhibitors to the scientific community. Initially featuring a robust set of tools for data extraction, analysis, and prediction from the ChEMBL database, Loki aims to expand its capabilities to serve as a central resource for cheminformatics research.

## Features
- **SQL Queries**: Robust SQL scripts to retrieve kinase-related compounds with validated activity measurements from the ChEMBL database.
- **Data Cleansing**: Advanced Python scripts for post-processing to remove redundancies and identify salt-free compounds.
- **Molecular Descriptors**: Automated tools for calculating essential molecular descriptors such as molecular weight, LogP, and more using RDKit.
- **Molecular Clustering**: Sophisticated algorithms to group molecules based on structural similarities and visualize clusters.
- **Visualization Tools**: Comprehensive scripts to generate histograms and other visual representations of data distributions and molecular properties.

## Repository Structure

    Loki/
    ├── ChEMBL/ # Tools for querying and processing data from the ChEMBL database
    │   ├── 1_kinase_compounds.sql # SQL script for extracting kinase inhibitor data
    │   ├── 2_remove_redundance.py # Python script for data cleansing and redundancy removal
    │   ├── 3_descriptors.py # Calculates and saves molecular descriptors
    │   ├── 4_cluster_by_similarity.py # Groups molecules by structural similarity
    │   ├── 5_histogram.py # Generates histograms of molecular similarities and distances
    │   ├── README.md # General information and usage instructions
    │   └── chembl_nr_pkidb_descriptors.tsv # Descriptors data file
    ├── MLmodel/ # Machine Learning models for predicting kinase ligand group
    │   ├── 1_check_data_balance.py # Analyzes class balance and generates visualizations
    │   ├── 2_ml_input.py # Preprocesses data to generate features suitable for ML models
    │   └── 3_modelling.py # Trains and evaluates various ML models to predict kinase groups
    └── README.md # Overview of the Loki project and navigation guide

## ChEMBL

The `ChEMBL` directory within the Loki repository provides a comprehensive set of SQL and Python scripts designed to interact with the ChEMBL database, specifically tailored to extract and process data regarding kinase inhibitors. These tools are intended to facilitate the retrieval, cleaning, and initial analysis of chemical data, which can then be used for more advanced computational chemistry and machine learning applications.

### Script Descriptions

- **1_kinase_compounds.sql**: SQL script dedicated to extracting detailed information on kinase inhibitors from the ChEMBL database. This script focuses on retrieving compound data with confirmed biological activities, providing a robust dataset for further analysis.

- **2_remove_redundance.py**: A Python script used to clean the extracted data by removing redundant entries and resolving issues like salt forms in compound structures, ensuring that the data integrity is maintained for scientific analysis.

- **3_descriptors.py**: Utilizes RDKit to calculate and store various molecular descriptors that are critical for the study of chemical compounds, especially those used in drug design and pharmacological assessments.

- **4_cluster_by_similarity.py**: Implements clustering algorithms to group kinase inhibitors based on their structural similarities, which helps in identifying unique or common features among different inhibitors.

- **5_histogram.py**: Generates histograms and other graphical representations to visualize the distribution of molecular properties and the results of similarity analyses among the compounds.

These tools are designed not only to support researchers in cheminformatics but also to serve as a foundation for more complex machine learning frameworks that predict the behavior or function of kinase inhibitors.

## MLmodel

The `MLmodel` folder in Loki contains a suite of Python scripts designed to facilitate the building and evaluation of machine learning models that predict kinase inhibitor targets based on molecular data. These scripts are crafted to handle large datasets efficiently, perform robust data preprocessing, feature extraction, and deploy multiple machine learning algorithms for comparative analysis.

### Script Descriptions

- **1_check_data_balance.py**: This script assesses the balance of data classes within the dataset. It provides visualizations and statistical measures to ensure the data used in model training is well-distributed, reducing biases towards any particular class.

- **2_ml_input.py**: Prepares the dataset for machine learning by converting SMILES strings to molecular fingerprints, splitting the dataset into training and testing sets, and applying various preprocessing methods. It ensures that the data is ready for effective model training.

- **3_modelling.py**: Implements multiple machine learning algorithms to evaluate their performance on the dataset. This script includes models such as Random Forests, SVM, and Neural Networks. It provides tools for cross-validation, performance evaluation, and the ability to save the best-performing models for further use.

## Getting Started
To get started with **Loki**, follow the setup instructions within each directory to properly configure and utilize the tools provided. As Loki expands, it will transition from a tool repository to a full-fledged server.

## Contributing
Contributions are essential for the growth and enhancement of **Loki**. Please refer to the CONTRIBUTING.md file for guidelines on contributing to this project.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
Special thanks to all the contributors and researchers whose insights and feedback have been invaluable in shaping **Loki** into a pivotal resource for the scientific community.


### Documentation

Each script is well-documented with comments explaining the steps taken and the rationale behind each decision. For a deeper understanding, refer to the inline comments within each script.

