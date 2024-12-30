import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from cuml.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import seaborn as sns

def kmeans(X_train, y_train, X_test, y_test, classes, save_model_path="./output/kmeans/kmeans_model.pkl", max_iter=130):
    if not isinstance(X_train, np.ndarray) or not isinstance(X_test, np.ndarray):
        raise ValueError("Input data should be in NumPy format for cuML KMeans.")

    print("Training K-Means on GPU ...")
    losses = []
    accuracies = []

    for i in range(1, max_iter + 1):
        model = KMeans(n_clusters=len(classes), init="k-means++", max_iter=i, random_state=42)
        model.fit(X_train)
        losses.append(model.inertia_)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        print(f"Iteration {i}, Loss (Inertia): {model.inertia_:.4f}, Accuracy: {acc:.4f}")

    final_model = KMeans(n_clusters=len(classes), init="k-means++", max_iter=max_iter, random_state=42)
    final_model.fit(X_train)
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    with open(save_model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"Final model saved to {save_model_path}")

    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker='o', label="Loss (Inertia)")
    plt.title("K-Means Training Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Inertia (Loss)")
    plt.legend()
    plt.savefig("./output/kmeans/kmeans_loss_curve.png")
    plt.close()
    print("Loss curve saved to ./output/kmeans/kmeans_loss_curve.png")

    plt.figure()
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', label="Accuracy", color="orange")
    plt.title("K-Means Accuracy Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("./output/kmeans/kmeans_accuracy_curve.png")
    plt.close()
    print("Accuracy curve saved to ./output/kmeans/kmeans_accuracy_curve.png")

    y_pred = final_model.predict(X_test)
    evaluate_model(y_test, y_pred, "K-Means", classes)

def evaluate_model(y_test, y_pred, model_name, classes):
    y_test = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
    y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"Model: {model_name}")
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"./output/kmeans/{model_name.replace(' ', '_')}_confusion_matrix.png")
    plt.close()
    print(f"Confusion matrix saved to ./output/kmeans/{model_name.replace(' ', '_')}_confusion_matrix.png")
