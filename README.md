# Loki (Look On Kinase Inhibitors)

## Overview
**Loki** is not just a toolkit, but a comprehensive server designed to provide valuable information on kinase inhibitors to the scientific community. Initially featuring a robust set of tools for data extraction, analysis, and prediction from the ChEMBL database, Loki aims to expand its capabilities to serve as a central resource for cheminformatics research.

## Features
- **SQL Queries**: Robust SQL scripts to retrieve kinase-related compounds with validated activity measurements from the ChEMBL database.
- **Data Cleansing**: Advanced Python scripts for post-processing to remove redundancies and identify salt-free compounds.
- **Molecular Descriptors**: Automated tools for calculating essential molecular descriptors such as molecular weight, LogP, and more using RDKit.
- **Molecular Clustering**: Sophisticated algorithms to group molecules based on structural similarities and visualize clusters.
- **Visualization Tools**: Comprehensive scripts to generate histograms and other visual representations of data distributions and molecular properties.

## Repository Structure
Loki/
│
├── ChEMBL/ # Initial tools for querying and processing data from the ChEMBL database
│ ├── kinase_compounds.sql # SQL script for extracting kinase inhibitor data
│ ├── remove_redundance.py # Python script for data cleansing and redundancy removal
│ └── ...
│
├── Descriptors/ # Scripts for calculating molecular descriptors
│ ├── descriptors.py # Calculates and saves molecular descriptors
│ └── ...
│
├── Clustering/ # Scripts for molecular clustering
│ ├── cluster_by_similarity.py # Groups molecules by structural similarity
│ └── ...
│
└── Visualization/ # Visualization scripts
├── histogram.py # Generates histograms of molecular similarities and distances
└── ...


## Getting Started
To get started with **Loki**, follow the setup instructions within each directory to properly configure and utilize the tools provided. As Loki expands, it will transition from a tool repository to a full-fledged server.

## Contributing
Contributions are essential for the growth and enhancement of **Loki**. Please refer to the CONTRIBUTING.md file for guidelines on contributing to this project.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
Special thanks to all the contributors and researchers whose insights and feedback have been invaluable in shaping **Loki** into a pivotal resource for the scientific community.
