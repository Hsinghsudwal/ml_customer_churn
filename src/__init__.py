from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np


def train_and_evaluate_models(X, y):
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=42
        ),
        "LightGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        "SVM": SVC(kernel="linear", probability=True, random_state=42),
    }

    best_model = None
    best_accuracy = 0

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy}")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name

    # Ensemble Voting Classifier
    ensemble_model = VotingClassifier(
        estimators=[
            ("rf", models["Random Forest"]),
            ("xgb", models["XGBoost"]),
            ("lgbm", models["LightGBM"]),
            ("cat", models["CatBoost"]),
            ("svm", models["SVM"]),
        ],
        voting="hard",
    )

    print("\nTraining Ensemble Model...")
    ensemble_model.fit(X_train, y_train)
    y_pred = ensemble_model.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, y_pred)

    print("\nEnsemble Model Results:")
    print(f"Accuracy: {ensemble_accuracy}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Check if ensemble model is the best
    if ensemble_accuracy > best_accuracy:
        best_accuracy = ensemble_accuracy
        best_model = "Ensemble Model"

    return best_model, best_accuracy


# Example usage with dummy data
if __name__ == "__main__":
    # Generate sample data
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )

    best_model, best_accuracy = train_and_evaluate_models(X, y)
    print(f"\nBest Model: {best_model} with Accuracy: {best_accuracy}")


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np


def train_and_evaluate_models(X, y):
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models with hyperparameter tuning using GridSearchCV
    param_grid = {
        "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
        "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
        "LightGBM": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
        "CatBoost": {"iterations": [100, 200], "learning_rate": [0.01, 0.1]},
        "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    }

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=42
        ),
        "LightGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }

    best_model = None
    best_accuracy = 0

    # Train and evaluate each model with GridSearchCV
    for name, model in models.items():
        print(f"\nTuning {name}...")
        grid_search = GridSearchCV(
            model, param_grid[name], cv=3, scoring="accuracy", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model_instance = grid_search.best_estimator_
        y_pred = best_model_instance.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n{name} Results:")
        print(f"Best Params: {grid_search.best_params_}")
        print(f"Accuracy: {accuracy}")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name

    # Ensemble Voting Classifier with best models
    ensemble_model = VotingClassifier(
        estimators=[
            ("rf", models["Random Forest"]),
            ("xgb", models["XGBoost"]),
            ("lgbm", models["LightGBM"]),
            ("cat", models["CatBoost"]),
            ("svm", models["SVM"]),
        ],
        voting="hard",
    )

    print("\nTraining Ensemble Model...")
    ensemble_model.fit(X_train, y_train)
    y_pred = ensemble_model.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, y_pred)

    print("\nEnsemble Model Results:")
    print(f"Accuracy: {ensemble_accuracy}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Check if ensemble model is the best
    if ensemble_accuracy > best_accuracy:
        best_accuracy = ensemble_accuracy
        best_model = "Ensemble Model"

    return best_model, best_accuracy


# Example usage with dummy data
if __name__ == "__main__":
    # Generate sample data
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )

    best_model, best_accuracy = train_and_evaluate_models(X, y)
    print(f"\nBest Model: {best_model} with Accuracy: {best_accuracy}")
