import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from tqdm import tqdm  # Importando tqdm

class PlotCluster:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = pd.read_csv(self.filepath, sep='\t')

    def smiles_to_fp(self, smiles, radius=2, nBits=2048):
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits) if mol else None

    def prepare_data(self):
        # Adicionando tqdm para monitorar o progresso
        self.data['fingerprint'] = [self.smiles_to_fp(smiles) for smiles in tqdm(self.data['canonical_smiles'], desc='Converting SMILES to Fingerprints')]
        fp_list = list(self.data['fingerprint'])
        fp_array = np.array([list(fp) for fp in fp_list if fp is not None])
        return fp_array

    def perform_tsne(self, fp_array):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(fp_array)
        self.data['tsne-2d-one'] = tsne_results[:, 0]
        self.data['tsne-2d-two'] = tsne_results[:, 1]

    def plot_clusters(self):
        # Define color palette
        kinase_groups = self.data['kinase_group'].unique()
        colors = matplotlib.colormaps['tab20'](np.linspace(0, 1, len(kinase_groups)))
        color_map = {group: color for group, color in zip(kinase_groups, colors)}

        # Plot
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="kinase_group",
            palette=color_map,
            data=self.data,
            legend="full",
            alpha=1.0
        )
        plt.title('t-SNE colored by Kinase Groups')
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        plt.savefig('tsne_colored_by_kinase_group.png')
        plt.show()

    def plot_kinase_clusters_only(self):
        # Filter out non-kinase groups
        kinase_only_groups = ['Nuclear', 'Membrane receptor', 'Transferase', 'Other', 'Lyase']
        filtered_data = self.data[~self.data['kinase_group'].isin(kinase_only_groups)]

        # Define color palette
        kinase_groups = filtered_data['kinase_group'].unique()
        colors = matplotlib.colormaps['tab20'](np.linspace(0, 1, len(kinase_groups)))
        color_map = {group: color for group, color in zip(kinase_groups, colors)}

        # Plot
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="kinase_group",
            palette=color_map,
            data=filtered_data,
            legend="full",
            alpha=1.0
        )
        plt.title('t-SNE excluding Non-Kinase Groups')
        plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        plt.savefig('tsne_excluding_non_kinase_groups.png')
        plt.show()

        # Save the filtered data to a TSV file
        filtered_data.to_csv('chembl_cluster_only_kinase_groups.tsv', sep='\t', index=False)

    @staticmethod
    def main():
        plotter = PlotCluster('./chembl_cluster_hits.tsv')
        fp_array = plotter.prepare_data()
        plotter.perform_tsne(fp_array)
        plotter.plot_kinase_clusters_only()

if __name__ == "__main__":
    PlotCluster.main()
