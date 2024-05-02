import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
import concurrent.futures
import os
import pyarrow as pa
import pyarrow.parquet as pq

class FormatFileML:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        data = pd.read_csv(self.filepath, sep='\t')
        return data

    @staticmethod
    def smiles_to_fingerprints(smiles, radius=2, n_bits=2048):
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
        batch_size = 524288
        fingerprints = []

        for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
            batch = data.iloc[i:i + batch_size]
            batch_fingerprints = self.convert_smiles_batch(batch['canonical_smiles'].tolist())
            fingerprints.extend(batch_fingerprints)

        data['fingerprint'] = fingerprints
        # Encode labels as integers
        label_encoder = {label: idx for idx, label in enumerate(data['kinase_group'].unique())}
        data['indexedLabel'] = data['kinase_group'].map(label_encoder)

        return data

    @staticmethod
    def split_data(data):
        X = np.array(data['fingerprint'].tolist())
        y = data['indexedLabel'].values  # Use indexedLabel as target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def save_to_parquet(X, y, filepath):
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = y
        table = pa.Table.from_pandas(df)
        pq.write_table(table, filepath)

def main():
    # Initialize formatter with the path to the dataset
    formatter = FormatFileML('../filtered_dataset.tsv')

    # Load data
    data = formatter.load_data()

    # Preprocess data
    data = formatter.preprocess_data(data)

    # Split data
    X_train, X_test, y_train, y_test = formatter.split_data(data)

    # Save the split data in Parquet format
    formatter.save_to_parquet(X_train, y_train, 'train_data.parquet')

if __name__ == '__main__':
    main()
