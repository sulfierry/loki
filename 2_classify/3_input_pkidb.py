import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def load_data(filepath):
    """ Load data from a TSV file. """
    return pd.read_csv(filepath, sep='\t')

def smiles_to_fingerprints(smiles, radius=2, n_bits=2048):
    """ Convert SMILES string to a Morgan fingerprint. """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=int)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits).ToList()

def preprocess_data(data, smiles_column='Canonical_Smiles'):
    """ Preprocess data by converting SMILES to fingerprints. """
    data['fingerprint'] = data[smiles_column].apply(smiles_to_fingerprints)
    X = np.array(data['fingerprint'].tolist())
    if 'Kinase_Group' in data.columns:
        y = data['Kinase_Group'].values
        return X, y
    return X, None

def main():
    # Load test data and model
    test_data_path = 'new_test_data.tsv'
    model_path = 'top_model_1.joblib'

    data = load_data(test_data_path)
    X_test, y_true = preprocess_data(data)

    # Load the trained model
    model = joblib.load(model_path)

    # Predict kinase groups
    y_pred = model.predict(X_test)

    # Display predictions for each molecule
    predictions = pd.DataFrame({
        'SMILES': data['Canonical_Smiles'],
        'Predicted_Group': y_pred
    })
    print(predictions)

    # If true labels are available, evaluate the model
    if y_true is not None:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

    # Save predictions to a file
    predictions.to_csv('predicted_kinase_groups.csv', index=False)

if __name__ == '__main__':
    main()
