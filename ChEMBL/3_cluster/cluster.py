import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from sklearn.manifold import TSNE
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, DataStructs
from matplotlib.colors import Normalize

class MoleculeClusterer:

    def __init__(self, smiles_file_path):
        self.smiles_file_path = smiles_file_path
        self.data = None
        self.fingerprints = []

    def load_data(self, smile_column):
        self.data = pd.read_csv(self.smiles_file_path, sep='\t')
        self.data = self.data.dropna(subset=[smile_column])

    def cluster_by_similarity(self, threshold=0.8):
        num_fps = len(self.fingerprints)
        clusters = []
        visited = set()

        for i in range(num_fps):
            if i in visited:
                continue

            cluster = [i]
            for j in range(i + 1, num_fps):
                if j not in visited:
                    similarity = DataStructs.TanimotoSimilarity(self.fingerprints[i], self.fingerprints[j])
                    if similarity >= threshold:
                        cluster.append(j)
                        visited.add(j)

            clusters.append(cluster)

        return clusters

    @staticmethod
    def smiles_to_fingerprint(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2) if mol else None


    def parallel_generate_fingerprints(self, smile_column, batch_size):
        num_cpus = os.cpu_count() // 2  # Ajuste para usar metade dos CPUs disponíveis para evitar sobrecarga
        smiles_list = self.data[smile_column].tolist()
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            results = list(tqdm(executor.map(self.smiles_to_fingerprint, smiles_list, chunksize=batch_size), total=len(smiles_list)))

        # Filtrar resultados não None e limpar memória
        self.fingerprints = [fp for fp in results if fp is not None]
        del results  # Libera a memória dos resultados imediatamente

    def save_clustered_data(self, output_file_path):
        self.data.to_csv(output_file_path, sep='\t', index=False)

    def save_state(self, file_path):
        state = {
            'data': self.data,
            'fingerprints': self.fingerprints
        }
        with open(file_path, 'wb') as file:
            pickle.dump(state, file)

    def load_state(self, file_path):
        with open(file_path, 'rb') as file:
            state = pickle.load(file)
            self.data = state['data']
            self.fingerprints = state['fingerprints']

    def calculate_tsne(self):
        fingerprint_array = np.array([fp for fp in tqdm(self.fingerprints, desc='Processing Fingerprints') if fp is not None])
        tsne = TSNE(n_components=2, random_state=42, learning_rate=2000.0, init='pca')
        tsne_results = tsne.fit_transform(fingerprint_array)
        return tsne_results

    def plot_tsne(self, tsne_results, threshold):
        plt.figure(figsize=(12, 8))
        cluster_counts = self.data['kinase_group'].value_counts()
        unique_groups = self.data['kinase_group'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_groups)))
        group_color_dict = dict(zip(unique_groups, colors))

        for group, color in group_color_dict.items():
            if cluster_counts[group] >= threshold:
                idxs = self.data[self.data['kinase_group'] == group].index
                valid_idxs = idxs[idxs < len(tsne_results)]  # Certifique-se de que os índices são válidos
                plt.scatter(tsne_results[valid_idxs, 0], tsne_results[valid_idxs, 1], label=group, color=color, alpha=0.5)

        plt.title('t-SNE clustering colored by kinase group')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(fontsize='small')
        plt.savefig('tsne_colored_by_kinase_group.png')
        plt.show()

    def plot_cluster_size_distribution(self, threshold):
        cluster_sizes = self.data['cluster_id'].value_counts()
        filtered_cluster_sizes = cluster_sizes[cluster_sizes >= threshold]

        plt.figure(figsize=(10, 6))
        plt.hist(filtered_cluster_sizes, bins=range(threshold, filtered_cluster_sizes.max() + 1), alpha=0.7, color='blue', edgecolor='black')
        plt.title('Smiles distribution per cluster')
        plt.xlabel('Number of smiles per cluster')
        plt.ylabel('Cluster count')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig('cluster_size_distribution.png')
        plt.show()


    def save_clusters_as_tsv(self, threshold, smile_column, cluster_hits_file):
        os.makedirs('./clusters', exist_ok=True)
        self.data['molecular_weight'] = self.data[smile_column].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)) if x else None)

        output_columns = [
            'molregno', 'target_kinase', 'canonical_smiles', 'standard_value', 'standard_type',
            'pchembl_value', 'compound_name', 'molecular_weight', 'cluster_id', 'kinase_group', 'count_kinase_group'
        ]

        cluster_hits = []

        for cluster_id in set(self.data['cluster_id']):
            cluster_data = self.data[self.data['cluster_id'] == cluster_id]
            cluster_data_sorted = cluster_data.sort_values('molecular_weight')

            if len(cluster_data_sorted) >= threshold:
                cluster_data_to_save = cluster_data_sorted[output_columns]
                cluster_data_to_save.to_csv(f'./clusters/cluster_{cluster_id}.tsv', sep='\t', index=False)
                cluster_hits.append(cluster_data_sorted.iloc[0][output_columns])

        cluster_hits_df = pd.DataFrame(cluster_hits, columns=output_columns)
        cluster_hits_df.to_csv(cluster_hits_file, sep='\t', index=False)

def run(smiles_file_path, output_file_path, state_file_path, tanimoto_threshold, cluster_size_threshold, smile_column, batch_size):
    clusterer = MoleculeClusterer(smiles_file_path)

    try:
        clusterer.load_state(state_file_path)
        print("Estado carregado com sucesso.")
    except FileNotFoundError:
        print("Nenhum estado salvo encontrado. Iniciando processamento do zero.")
        clusterer.load_data(smile_column)
        clusterer.parallel_generate_fingerprints(smile_column, batch_size)
        clusterer.save_state(state_file_path)
    except Exception as e:
        print(f"Erro ao carregar o estado: {e}")
        return

    if clusterer.fingerprints:
        tsne_results = clusterer.calculate_tsne()
        clusters = clusterer.cluster_by_similarity(threshold=tanimoto_threshold)

        cluster_ids = [None] * len(clusterer.data)  # Assegurar que esta lista tem o tamanho correto
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                if idx < len(cluster_ids):  # Verificação de segurança
                    cluster_ids[idx] = cluster_id
        clusterer.data['cluster_id'] = cluster_ids

        clusterer.save_clusters_as_tsv(cluster_size_threshold, smile_column, './chembl_cluster_hits.tsv')
        #clusterer.plot_tsne(tsne_results, cluster_size_threshold)
        #clusterer.plot_cluster_size_distribution(cluster_size_threshold)

        #clusterer.save_state(state_file_path)
    else:
        print("Nenhum fingerprint válido foi encontrado.")


def main():
    smiles_file_path = '../1_remove_redundance/nr_kinase_all_compounds_salt_free.tsv'
    output_file_path = './clustered_smiles.tsv'
    state_file_path = './molecule_clusterer_state.pkl'
    smile_column = 'canonical_smiles'
    tanimoto_threshold = 0.8
    cluster_size_threshold = 3  # numero minimo de moleculas por cluster
    batch_size = 10240  # tamanho do lote para processamento paralelo

    run(smiles_file_path, output_file_path, state_file_path, tanimoto_threshold, cluster_size_threshold, smile_column, batch_size)

if __name__ == "__main__":
    main()
