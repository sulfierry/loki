import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer, balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from imblearn.metrics import geometric_mean_score  # Importar a partir do imbalanced-learn
from sklearn.pipeline import Pipeline
from sklearn.utils import parallel_backend

# Importing various classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

class Classifiers:
    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()
        self.kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def load_data(self):
        loader = np.load(self.data_path, allow_pickle=True)
        self.X_train, self.X_val, self.X_test = loader['X_train'], loader['X_val'], loader['X_test']
        self.y_train, self.y_val, self.y_test = loader['y_train'], loader['y_val'], loader['y_test']

    def evaluate_model(self, model, name):
        metrics = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall': make_scorer(recall_score, average='weighted', zero_division=0),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0),
            'roc_auc_ovr': 'roc_auc_ovr',
            'balanced_accuracy': 'balanced_accuracy',
            'geometric_mean': make_scorer(geometric_mean_score, average='weighted')  # Usando a métrica correta
        }
        with parallel_backend('loky', n_jobs=2):
            scores = cross_validate(model, self.X_train, self.y_train, cv=self.kfold,
                                    scoring=metrics, return_train_score=False,
                                    return_estimator=True, n_jobs=2)
        results = {metric: np.mean(scores[f'test_{metric}']) for metric in metrics}
        return results, scores['estimator'][-1]  # Return the last fitted estimator

    def train_and_evaluate(self):
        results = []
        model_details = []
        for name, clf in tqdm([
            ("Nearest Neighbors", KNeighborsClassifier(3)),
            ("Linear SVM", SVC(kernel="linear", C=0.025, probability=True)),
            ("RBF SVM", SVC(gamma=2, C=1, probability=True)),
            ("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))),
            ("Decision Tree", DecisionTreeClassifier(max_depth=5)),
            ("Random Forest", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
            ("Neural Net", MLPClassifier(alpha=1, max_iter=1000)),
            ("AdaBoost", AdaBoostClassifier()),
            ("Naive Bayes", GaussianNB()),
            ("QDA", QuadraticDiscriminantAnalysis()),
            ("Logistic Regression", LogisticRegression(max_iter=1000)),
            ("Gradient Boosting", GradientBoostingClassifier()),
        ], desc="Processing classifiers"):
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', clf)
            ])
            result, fitted_model = self.evaluate_model(model, name)
            results.append((name, result))
            model_details.append((name, result, fitted_model))

        # Save the top 3 models based on a chosen metric, e.g., accuracy
        top_models = sorted(model_details, key=lambda x: x[1]['accuracy'], reverse=True)[:3]
        for i, (name, _, model) in enumerate(top_models):
            joblib.dump(model, f'top_model_{i + 1}_{name}.joblib')
            print(f"Saved {name} as 'top_model_{i + 1}_{name}.joblib'")
        return results

def plot_metrics(results):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc_ovr', 'geometric_mean']
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        names = [res[0] for res in results]
        values = [res[1][metric] for res in results if metric in res[1]]
        ax.barh(names, values, color='skyblue')
        ax.set_title(f'{metric.title()} Score')
        ax.set_xlabel('Score')
        ax.set_xlim(0, 1.0)
    plt.tight_layout()
    plt.show()

def main():
    classifier = Classifiers('split_data.npz')
    results = classifier.train_and_evaluate()
    print("\nEvaluation Results:")
    for name, result in results:
        print(f"\n{name} Classifier Metrics:")
        for metric, value in result.items():
            print(f"  {metric}: {value:.2f}")

    plot_metrics(results)

if __name__ == '__main__':
    main()
