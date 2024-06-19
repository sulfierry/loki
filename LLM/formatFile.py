import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import pyarrow as pa
import pyarrow.parquet as pq

INPUT_FILE = './filtered_dataset.tsv'
BATCH_SIZE = 249856

class FormatFileML:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        # Load only the necessary columns
        columns = ['chembl_id', 'molregno', 'target_kinase', 'canonical_smiles',
                   'standard_value', 'standard_type', 'kinase_group']
        data = pd.read_csv(self.filepath, sep='\t', usecols=columns)

        # Clean data
        # Drop rows where any element is NaN
        data.dropna(inplace=True)

        # Ensure 'standard_value' does not contain extreme values or placeholders
        data['standard_value'] = data['standard_value'].replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
        if data['standard_value'].isna().any():
            data['standard_value'] = data['standard_value'].fillna(data['standard_value'].median())  # Fill NaNs with median

        return data

    def preprocess_data(self, data):
        # For ChemBERTa, we just need the SMILES and the target labels
        processed_data = data[['canonical_smiles', 'kinase_group']].copy()
        processed_data.rename(columns={'kinase_group': 'target'}, inplace=True)
        return processed_data

    @staticmethod
    def split_data(data):
        X = data['canonical_smiles'].values
        y = data['target'].values
        smiles_train, smiles_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return smiles_train, smiles_test, y_train, y_test

    @staticmethod
    def save_to_parquet(data, filepath):
        table = pa.Table.from_pandas(data)
        pq.write_table(table, filepath)

def main():
    # Initialize formatter with the path to the dataset
    formatter = FormatFileML(INPUT_FILE)

    # Load data
    data = formatter.load_data()

    # Preprocess data
    data = formatter.preprocess_data(data)

    # Split data
    smiles_train, smiles_test, y_train, y_test = formatter.split_data(data)

    # Convert to DataFrame to save in Parquet
    df_train = pd.DataFrame({'canonical_smiles': smiles_train, 'target': y_train})
    df_test = pd.DataFrame({'canonical_smiles': smiles_test, 'target': y_test})

    # Optionally save the split data in .npz and Parquet formats
    np.savez('split_data.npz', smiles_train=smiles_train, smiles_test=smiles_test, y_train=y_train, y_test=y_test)
    formatter.save_to_parquet(df_train, 'train_data.parquet')
    formatter.save_to_parquet(df_test, 'test_data.parquet')

if __name__ == '__main__':
    main()
