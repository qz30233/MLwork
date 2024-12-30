from cuml.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

def random_forest(X_train, y_train, X_test, y_test, classes, save_model_path="./output/Random_Forest/random_forest_model.pkl"):
    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []

    print("Training Random Forest on GPU...")
    for n_estimators in range(10, 101, 10):
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)
        train_acc.append(accuracy_score(y_train, y_train_pred))
        train_loss.append(log_loss(y_train, y_train_proba))

        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)
        test_acc.append(accuracy_score(y_test, y_test_pred))
        test_loss.append(log_loss(y_test, y_test_proba))

    plot_curves(train_acc, test_acc, train_loss, test_loss)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    with open(save_model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {save_model_path}")

    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "Random Forest", classes)

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
    
    output_dir = "./output/Random_Forest/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_curves(train_acc, test_acc, train_loss, test_loss):

    output_dir = "./output/Random_Forest/"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(range(10, 101, 10), train_acc, label="Training Accuracy", marker="o")
    plt.plot(range(10, 101, 10), test_acc, label="Validation Accuracy", marker="o")
    plt.title("Accuracy Curve During Training")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Accuracy")
    plt.legend()
    acc_path = os.path.join(output_dir, "accuracy_curve.png")
    plt.savefig(acc_path)
    plt.close()
    print(f"Accuracy curve saved to {acc_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(10, 101, 10), train_loss, label="Training Loss", marker="o")
    plt.plot(range(10, 101, 10), test_loss, label="Validation Loss", marker="o")
    plt.title("Loss Curve During Training")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Loss")
    plt.legend()
    loss_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Loss curve saved to {loss_path}")
