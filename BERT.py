import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
import matplotlib.pyplot as plt
import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset

file_path = 'bbc_data.csv'
df = pd.read_csv(file_path)
df["text"] = df["text"].astype(str)

# 将文本标签映射为数值编码
label_mapping = {label: idx for idx, label in enumerate(df["label"].unique())}
df["label"] = df["label"].map(label_mapping)

df = df.dropna()
train_df, test_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)

# 加载 BERT 分词器和模型
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_mapping))

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 设置数据对齐器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# 定义评价指标
def compute_metrics(pred):
    predictions = pred.predictions.argmax(axis=-1)
    labels = pred.label_ids
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# 使用 Trainer 进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

train_result = trainer.train()

train_loss=[]
train_metrics = train_result.metrics
train_loss.append(train_metrics.get("train_loss", []))
print("train_metrics")
print(train_metrics)
print("train_loss")
print(train_loss)

#  模型评估
test_accuracy=[]
eval_metrics = trainer.evaluate()
test_accuracy.append(eval_metrics.get("eval_accuracy", []))
test_loss = []
test_loss.append(eval_metrics.get("eval_loss", []))
print("eval_metrics")
print(eval_metrics)
print("test_accuracy")
print(test_accuracy)
print("test_loss")
print(test_loss)

predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(axis=-1)

# 将数值标签转换回文本标签
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
test_df["label"] = test_df["label"].map(reverse_label_mapping)
predicted_labels_text = [reverse_label_mapping[label] for label in predicted_labels]

labels_numeric = test_dataset["label"]
mae = mean_absolute_error(labels_numeric, predicted_labels)
mse = mean_squared_error(labels_numeric, predicted_labels)
rmse = np.sqrt(mse)

print("分类报告：")
print(classification_report(test_df["label"], predicted_labels_text, target_names=reverse_label_mapping.values()))
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")