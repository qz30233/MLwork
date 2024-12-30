import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from logistic_regression import logistic_regression
from svm import svm
from random_forest import random_forest
from improved_random_forest import improved_random_forest
from kmeans import kmeans

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    return train_loader, test_loader, train_dataset.classes

def dataset_to_numpy(data_loader):
    images, labels = [], []
    for inputs, targets in data_loader:
        images.append(inputs.view(inputs.size(0), -1).numpy())
        labels.append(targets.numpy())
    return np.vstack(images), np.concatenate(labels)


if __name__ == "__main__":
    print("Loading data...")
    train_loader, test_loader, classes = load_data()

    print("Converting dataset to NumPy format...")
    X_train, y_train = dataset_to_numpy(train_loader)
    X_test, y_test = dataset_to_numpy(test_loader)

    # 运行各模型
    #print("Running Logistic Regression...")
    #logistic_regression(X_train, y_train, X_test, y_test, classes)

    #print("Running SVM...")
    #svm(X_train, y_train, X_test, y_test, classes,50)

    #print("Running Random Forest...")
    #random_forest(X_train, y_train, X_test, y_test, classes)

    #print("Running K-Means...")
    #kmeans(X_train, y_train, X_test, y_test, classes)
    
    print("Running improved_random_forest...")
    improved_random_forest(X_train, y_train, X_test, y_test, classes)

    print(f"Results saved in {output_dir}")
