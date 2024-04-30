import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
import concurrent.futures
import os

def load_data(filepath):
    return pd.read_csv(filepath, sep='\t')

def smiles_to_fingerprints(smiles, radius=2, n_bits=2048):
    if pd.isna(smiles):
        return [0] * n_bits  # Retorna um fingerprint nulo se o SMILES é NaN
    mol = Chem.MolFromSmiles(str(smiles))  # Garante que o input para MolFromSmiles seja uma string
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits).ToList()
    return [0] * n_bits

def convert_smiles_batch(smiles_list, radius=2, n_bits=2048):
    """ Convert a batch of SMILES to fingerprints using multiple processes. """
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(smiles_to_fingerprints, smiles_list,
                                         [radius] * len(smiles_list), [n_bits] * len(smiles_list)),
                           total=len(smiles_list), desc="Converting SMILES"))
    return results

def preprocess_data(data):
    batch_size = 8192
    fingerprints = []

    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch = data.iloc[i:i + batch_size]
        batch_fingerprints = convert_smiles_batch(batch['canonical_smiles'].tolist())
        fingerprints.extend(batch_fingerprints)

    data['fingerprint'] = fingerprints

    # Prepare labels based on 'standard_value'
    threshold_active = data['standard_value'].quantile(0.50)
    threshold_inactive = data['standard_value'].quantile(0.75)
    data['label'] = data['standard_value'].apply(lambda x: 'active' if x <= threshold_active else ('inactive' if x >= threshold_inactive else 'intermediate'))

    # Remove intermediate compounds
    data = data[data['label'] != 'intermediate']

    return data


def split_data(data):
    # Convert list of fingerprint lists to numpy array for ML models
    X = np.array(data['fingerprint'].tolist())
    y = data['label'].values

    # Splitting the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% training, 20% for split into val and test
    #X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Splitting the 20% into 10% validation and 10% test

    #return X_train, X_val, X_test, y_train, y_val, y_test
    return X_train, X_test, y_train, y_test



def main():
    # Load data
    filepath = '../1_remove_redundance/nr_kinase_all_compounds_salt_free_ver2.tsv'  # Adjust as needed
    data = load_data(filepath)

    # Preprocess data
    data = preprocess_data(data)

    # Split data
    #X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    X_train, X_test, y_train, y_test = split_data(data)

    # Optionally save the split data
    #np.savez('split_data.npz', X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)
    np.savez('split_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

if __name__ == '__main__':
    main()

