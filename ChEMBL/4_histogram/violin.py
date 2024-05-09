from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.spatial.distance import cosine
import seaborn as sns
from tqdm import tqdm

class ViolinPlot:

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

    def process_in_batches(self, data, process_function, metric):
        results = []
        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i+self.batch_size]
                futures.append(executor.submit(self.process_batch, batch, process_function, metric))
            for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing {metric}'):
                results.extend(future.result())
        # Normalize distances to 0-1 range
        if metric in ['hamming', 'manhattan']:
            max_val = max(results)
            min_val = min(results)
            results = [(x - min_val) / (max_val - min_val) for x in results]
        return results

    def create_violin_plots(self):
        data = pd.read_csv(self.file_path, sep='\t')
        data['fingerprints'] = data['canonical_smiles'].apply(self.smiles_to_fingerprint)
        valid_data = data.dropna(subset=['fingerprints'])
        valid_fps = valid_data['fingerprints'].tolist()

        similarity_metrics = ['tanimoto', 'dice', 'cosine']
        distance_metrics = ['hamming', 'manhattan']
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
        all_similarities = {metric: self.process_in_batches(valid_fps, self.calculate_similarity, metric) for metric in similarity_metrics}
        all_distances = {metric: self.process_in_batches(valid_fps, self.calculate_distance, metric) for metric in distance_metrics}

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        color_index = 0

        # Similarity plots
        for i, (metric, values) in enumerate(all_similarities.items()):
            sns.violinplot(data=values, ax=axs[0, i], color=colors[color_index])
            axs[0, i].set_title(f'Similarity - {metric.capitalize()}')
            color_index += 1

        # Distance plots
        for i, (metric, values) in enumerate(all_distances.items()):
            sns.violinplot(data=values, ax=axs[1, i], color=colors[color_index])
            axs[1, i].set_title(f'Distance - {metric.capitalize()}')
            color_index += 1

        # pChEMBL Value plot
        sns.violinplot(data=valid_data['pchembl_value'], ax=axs[1, 2], color=colors[color_index])
        axs[1, 2].set_title('pChEMBL Value Distribution')

        plt.tight_layout()
        plt.savefig('violin_plots_all.png')
        plt.close(fig)

if __name__ == "__main__":
    violin_plot = ViolinPlot('../3_cluster/chembl_cluster_hits.tsv')
    violin_plot.create_violin_plots()
