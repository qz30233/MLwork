import pandas as pd
from scipy.io import mmread
import os

mtx_file = "bbc.mtx"
terms_file = "bbc.terms"
docs_file = "bbc.docs"
classes_file = "bbc.classes"
bbc_folder = "./bbc/"

# 读取文件
terms = []
with open(terms_file, 'r', encoding='utf-8') as f:
    terms = [line.strip() for line in f]

docs = []
with open(docs_file, 'r', encoding='utf-8') as f:
    docs = [line.strip() for line in f]

classes = []
with open(classes_file, 'r', encoding='utf-8') as f:
    classes = [line.strip() for line in f]

# 读取稀疏矩阵
sparse_matrix = mmread(mtx_file).tocsc()

# 加载BBC文件夹的文档
data = []
for category in os.listdir(bbc_folder):
    category_folder = os.path.join(bbc_folder, category)
    if not os.path.isdir(category_folder):
        continue
    for file_name in os.listdir(category_folder):
        file_path = os.path.join(category_folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            data.append({"text": text, "label": category})

df = pd.DataFrame(data)
print(df.head())
df.to_csv("bbc_data.csv", index=False, encoding="utf-8")
