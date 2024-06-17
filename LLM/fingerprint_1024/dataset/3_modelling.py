import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
from imblearn.metrics import geometric_mean_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

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
# 32768
class Classifiers:
    def __init__(self, data_path, batch_size=32768):
        self.data_path = data_path
        self.batch_size = batch_size
        self.load_data()
        self.kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def load_data(self):
        loader = np.load(self.data_path, allow_pickle=True)
        self.X_train, self.X_test = loader['X_train'], loader['X_test']
        self.y_train, self.y_test = loader['y_train'], loader['y_test']

    def evaluate_model(self, model, name):
        metrics = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'balanced_accuracy': balanced_accuracy_score,
            'geometric_mean': geometric_mean_score
        }

        results = {metric: [] for metric in metrics}
        for train_idx, test_idx in self.kfold.split(self.X_train, self.y_train):
            X_train_fold, y_train_fold = self.X_train[train_idx], self.y_train[train_idx]
            X_test_fold, y_test_fold = self.X_train[test_idx], self.y_train[test_idx]

            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_test_fold)
            for metric, func in metrics.items():
                if metric in ['precision', 'recall', 'f1', 'geometric_mean']:
                    score = func(y_test_fold, y_pred, average='macro', zero_division=0)
                else:
                    score = func(y_test_fold, y_pred)
                results[metric].append(score)

        averaged_results = {metric: np.mean(scores) for metric, scores in results.items()}
        return averaged_results, model

    def train_and_evaluate(self):
        results = []
        model_details = []
        classifiers = [
            ("Nearest Neighbors", KNeighborsClassifier(3)),
            ("Linear SVM", SVC(kernel="linear", C=0.025, probability=True)),
            ("RBF SVM", SVC(gamma=2, C=1, probability=True)),
            ("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))),
            ("Decision Tree", DecisionTreeClassifier(max_depth=5)),
            ("Random Forest", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
            ("Neural Net", MLPClassifier(alpha=1, max_iter=1000)),
            ("AdaBoost", AdaBoostClassifier()),
            ("Naive Bayes", GaussianNB(var_smoothing=1e-8)),
            ("QDA", QuadraticDiscriminantAnalysis()),
            ("Logistic Regression", LogisticRegression(max_iter=1000)),
            ("Gradient Boosting", GradientBoostingClassifier())
        ]
        for name, clf in tqdm(classifiers, desc="Processing classifiers"):
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', clf)
            ])
            result, fitted_model = self.evaluate_model(model, name)
            results.append((name, result))
            model_details.append((name, result, fitted_model))

        top_models = sorted(model_details, key=lambda x: x[1]['accuracy'], reverse=True)[:3]
        for i, (name, _, model) in enumerate(top_models):
            joblib.dump(model, f'top_model_{i + 1}_{name}.joblib')
            print(f"Saved {name} as 'top_model_{i + 1}_{name}.joblib'")
        return results

def plot_metrics(results):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy', 'geometric_mean']
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
    classifier = Classifiers('./split_data.npz')
    results = classifier.train_and_evaluate()
    print("\nEvaluation Results:")
    for name, result in results:
        print(f"\n{name} Classifier Metrics:")
        for metric, value in result.items():
            print(f"  {metric}: {value:.2f}")

    plot_metrics(results)

if __name__ == '__main__':
    main()
