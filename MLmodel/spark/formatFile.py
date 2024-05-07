import os
import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
from rdkit import Chem
import concurrent.futures
import pyarrow.parquet as pq
from rdkit.Chem import AllChem
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


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


    @staticmethod
    def smiles_to_fingerprints(smiles, radius=2, n_bits=1024):
        if pd.isna(smiles):
            return [0] * n_bits  # Returns a null fingerprint if SMILES is NaN
        mol = Chem.MolFromSmiles(str(smiles))
        if mol:
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits).ToList()
        return [0] * n_bits

    @staticmethod
    def convert_smiles_batch(smiles_list, radius=2, n_bits=2048):
        """ Convert a batch of SMILES to fingerprints using multiple processes. """
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(tqdm(executor.map(FormatFileML.smiles_to_fingerprints, smiles_list,
                                             [radius] * len(smiles_list), [n_bits] * len(smiles_list)),
                               total=len(smiles_list), desc="Converting SMILES"))
        return results

    def preprocess_data(self, data):
        batch_size = 65536
        fingerprints = []

        for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
            batch = data.iloc[i:i + batch_size]
            batch_fingerprints = self.convert_smiles_batch(batch['canonical_smiles'].tolist())
            fingerprints.extend(batch_fingerprints)

        # Convert list of fingerprints to DataFrame
        df_fingerprints = pd.DataFrame(fingerprints)

        # Apply MinMaxScaler to normalize the fingerprints
        scaler = MinMaxScaler()
        normalized_fingerprints = scaler.fit_transform(df_fingerprints)

        # Append normalized fingerprints back to the original DataFrame
        data['fingerprint'] = list(normalized_fingerprints)
        return data

    @staticmethod
    def split_data(data):
        X = np.array(data['fingerprint'].tolist())
        y = data['kinase_group'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def save_to_parquet(data, filepath):
        table = pa.Table.from_pandas(data)
        pq.write_table(table, filepath)

def main():
    # Initialize formatter with the path to the dataset
    formatter = FormatFileML('./filtered_dataset.tsv')

    # Load data
    data = formatter.load_data()

    # Preprocess data
    data = formatter.preprocess_data(data)

    # Split data
    X_train, X_test, y_train, y_test = formatter.split_data(data)

    # Convert to DataFrame to save in Parquet
    df_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
    df_train['target'] = y_train

    # Optionally save the split data in .npz and Parquet formats
    np.savez('split_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    formatter.save_to_parquet(df_train, 'train_data.parquet')

if __name__ == '__main__':
    main()
