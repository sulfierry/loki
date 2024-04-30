import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

class CheckBalance:
    def __init__(self, filepath, activity_threshold):
        self.filepath = filepath
        self.activity_threshold = activity_threshold
        self.data = self.load_data()

    def load_data(self):
        # Load only the necessary columns
        columns = ['chembl_id', 'molregno', 'target_kinase', 'canonical_smiles',
                   'standard_value', 'standard_type', 'kinase_group']
        return pd.read_csv(self.filepath, sep='\t', usecols=columns)

    def analyze_class_balance(self):
        # Prepare labels based on activity thresholds
        self.data['label'] = self.data['standard_value'].apply(lambda x: 'active' if x < self.activity_threshold else 'inactive')

        # Analyze class balance within each kinase group
        class_counts = self.data.groupby(['kinase_group', 'label']).size().unstack(fill_value=0)

        # Plotting the overall balance
        plt.figure(figsize=(10, 6))
        sns.countplot(x='label', data=self.data)
        plt.title('Overall Class Balance')
        plt.xlabel('Activity Class')
        plt.ylabel('Number of Compounds')
        plt.show()

        # Plotting the balance by kinase group
        class_counts.plot(kind='bar', stacked=True, figsize=(14, 8))
        plt.ylabel('Number of Compounds')
        plt.title('Class Balance by Kinase Group')
        plt.show()

        # Detailed distribution plot
        plt.figure(figsize=(14, 8))
        detailed_plot = sns.histplot(data=self.data, x='kinase_group', hue='label', multiple="dodge", shrink=.8)
        detailed_plot.set_yscale('log')
        sns.histplot(data=self.data, x='kinase_group', hue='label', multiple="dodge", shrink=.8)
        plt.title('Distribution of Activity Classes within Kinase Groups')
        plt.xticks(rotation=45)
        plt.xlabel('Kinase Group')
        plt.ylabel('Count')
        plt.show()

    def analyze_class_metrics(self):
        # Calculate class counts and ratios
        class_counts = self.data['label'].value_counts()
        class_ratio = class_counts['active'] / class_counts['inactive'] if 'active' in class_counts and 'inactive' in class_counts else float('inf')

        # Calculate entropy
        probability_distribution = class_counts / class_counts.sum()
        data_entropy = entropy(probability_distribution, base=2)

        # Calculate the coefficient of variation
        coeff_variation = np.std(class_counts) / np.mean(class_counts)

        print("Class Ratio:", class_ratio)
        print("Entropy of Distribution:", data_entropy)
        print("Coefficient of Variation:", coeff_variation)

def main():
    # Set activity threshold for nM (e.g., 10µM expressed as 10000 nM)
    ACTIVITY_THRESHOLD = 1000
    filepath = '../1_remove_redundance/nr_kinase_all_compounds_salt_free_ver2.tsv'  # Adjust as needed
    
    checker = CheckBalance(filepath, ACTIVITY_THRESHOLD)
    checker.analyze_class_balance()
    checker.analyze_class_metrics()

if __name__ == '__main__':
    main()
