"""
	O presente algoritmo compara todos os fingerprints moleculares uns contra os outros,
	o que é conhecido como uma abordagem "todos contra todos". Para cada fingerprint no
	conjunto de dados,ele calcula a similaridade ou a distância com todos os outros
	fingerprints. Este processo é repetido para cada fingerprint, resultando em um
	conjunto abrangente de medidas de similaridade e distância entre cada par de moléculas
	representadas pelos fingerprints.

	Essa abordagem pode ser computacionalmente intensiva, especialmente para conjuntos de dados
	grandes, porque o número de comparações cresce quadraticamente com o número de fingerprints.
	Por exemplo, se há N fingerprints, o número de comparações será N×(N−1)/2, o que pode ser um
	número muito grande para grandes conjuntos de dados. É por isso que o algoritmo processa os
	dados em lotes, para gerenciar o uso de memória e recursos de processamento.
"""



from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.spatial.distance import cosine
import seaborn as sns
from tqdm import tqdm

class Histogram:

    def __init__(self, file_path, batch_size=1024):
        self.file_path = file_path
        self.batch_size = batch_size

    @staticmethod
    def smiles_to_fingerprint(smiles, radius=2):
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius) if mol else None

    def process_batch(self, batch, process_function, metric):
        results = []
        for fp in batch:
            results.extend(process_function(fp, batch, metric))
        return results

    @staticmethod
    def calculate_similarity(fingerprint, fingerprints, similarity_metric):
        similarities = []
        for fp in fingerprints:
            if similarity_metric == 'cosine':
                arr1, arr2 = np.zeros((1,)), np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fingerprint, arr1)
                DataStructs.ConvertToNumpyArray(fp, arr2)
                similarity = 1 - cosine(arr1, arr2)
            elif similarity_metric == 'tanimoto':
                similarity = DataStructs.TanimotoSimilarity(fingerprint, fp)
            elif similarity_metric == 'dice':
                similarity = DataStructs.DiceSimilarity(fingerprint, fp)
            similarities.append(similarity)
        return similarities

    def process_in_batches(self, data, process_function, metric):
        results = []
        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i+self.batch_size]
                futures.append(executor.submit(self.process_batch, batch, process_function, metric))
            for future in tqdm(as_completed(futures), total=len(futures), desc='Processing all batches'):
                results.extend(future.result())
        return results

    @staticmethod
    def calculate_distance(fingerprint, fingerprints, distance_metric):
        distances = []
        for fp in fingerprints:
            if distance_metric == 'hamming':
                arr1, arr2 = np.zeros((1,)), np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fingerprint, arr1)
                DataStructs.ConvertToNumpyArray(fp, arr2)
                distance = np.mean(arr1 != arr2)
            elif distance_metric == 'manhattan':
                arr1, arr2 = np.zeros((1,)), np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fingerprint, arr1)
                DataStructs.ConvertToNumpyArray(fp, arr2)
                distance = np.sum(np.abs(arr1 - arr2))
            distances.append(distance)
        return distances

    def create_histograms(self):
        data = pd.read_csv(self.file_path, sep='\t')
        data['fingerprints'] = data['canonical_smiles'].apply(self.smiles_to_fingerprint)
        valid_data = data.dropna(subset=['fingerprints'])
        valid_fps = valid_data['fingerprints'].tolist()

        similarity_metrics = ['tanimoto', 'dice', 'cosine']
        distance_metrics = ['hamming', 'manhattan']
        all_similarities = {metric: self.process_in_batches(valid_fps, self.calculate_similarity, metric) for metric in similarity_metrics}
        all_distances = {metric: self.process_in_batches(valid_fps, self.calculate_distance, metric) for metric in distance_metrics}

        fig, axs = plt.subplots(3, 2, figsize=(13, 13))
        # Histograms
        for ax, (metric, values) in zip(axs[:, 0], all_similarities.items()):
            ax.hist(values, bins=50, color='skyblue', edgecolor='black')
            ax.set_title(f'Similarity - {metric.capitalize()} Histogram', fontsize=10)
            ax.set_ylabel('Frequency')

        for ax, (metric, values) in zip(axs[:, 1], all_distances.items()):
            ax.hist(values, bins=50, color='salmon', edgecolor='black')
            ax.set_title(f'Distance - {metric.capitalize()} Histogram', fontsize=10)
            ax.set_ylabel('Frequency')

        # Histogram for pchembl_value
        ax = axs[2, 1]
        ax.hist(data['pchembl_value'].dropna(), bins=20, color='green', edgecolor='black')
        ax.set_title('pChEMBL Value Distribution', fontsize=10)
        ax.set_ylabel('Frequency')
        ax.set_xlabel('pChEMBL Value')

        plt.tight_layout()
        plt.savefig('histogram_similarity_distance_pchembl.png')
        plt.close(fig)

if __name__ == "__main__":
    histogram = Histogram('../3_cluster/chembl_cluster_hits.tsv')
    histogram.create_histograms()

