from cuml.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import numpy as np

def improved_random_forest(X_train, y_train, X_test, y_test, classes, save_model_path="./output/weighted_random_forest/random_forest_model.pkl"):
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    param_distributions = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [10, 20, 30,None],
        "max_features": ["sqrt", "log2", None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }

    print("Optimizing hyperparameters with Randomized Search...")
    rf = RandomForestClassifier()
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=100,
        scoring="accuracy",
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    print(f"Best parameters: {random_search.best_params_}")

    with open(save_model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Model saved to {save_model_path}")

    y_pred = best_model.predict(X_test)
    evaluate_model(y_test, y_pred, "Weighted Random Forest", classes)

def evaluate_model(y_test, y_pred, model_name, classes):
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    print(f"Model: {model_name}")
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    output_dir = "./output/weighted_random_forest/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
