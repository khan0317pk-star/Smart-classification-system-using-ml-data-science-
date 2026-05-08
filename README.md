# Smart-classification-system-using-ml-data-science-


import os
import zipfile
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings('ignore')

zip_path = r"C:\Users\ABC\Downloads\archive.zip"

extract_folder = r"C:\Users\ABC\Downloads\kaggle_dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

print("ZIP file extracted successfully!")

files = os.listdir(extract_folder)

print(files)

csv_file = None

for file in files:
    if file.endswith(".csv"):
        csv_file = os.path.join(extract_folder, file)
        break

if csv_file is None:
    raise FileNotFoundError("No CSV file found!")

print(csv_file)

df = pd.read_csv(csv_file)

print(df.head())

print(df.shape)

print(df.columns)

print(df.info())

print(df.isnull().sum())

df.drop_duplicates(inplace=True)

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))

plt.figure(figsize=(12, 8))

sns.heatmap(df.corr(), cmap='coolwarm')

plt.show()

print(df.columns)

target_column = df.columns[-1]

print(target_column)

X = df.drop(target_column, axis=1)

y = df[target_column]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print(accuracy_score(y_test, lr_pred))

print(classification_report(y_test, lr_pred))

plt.figure(figsize=(6, 4))

sns.heatmap(
    confusion_matrix(y_test, lr_pred),
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.show()

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print(accuracy_score(y_test, dt_pred))

print(classification_report(y_test, dt_pred))

plt.figure(figsize=(6, 4))

sns.heatmap(
    confusion_matrix(y_test, dt_pred),
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.show()

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print(accuracy_score(y_test, rf_pred))

print(classification_report(y_test, rf_pred))

plt.figure(figsize=(6, 4))

sns.heatmap(
    confusion_matrix(y_test, rf_pred),
    annot=True,
    fmt='d',
    cmap='Reds'
)

plt.show()

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print(feature_importance.head(10))

plt.figure(figsize=(10, 6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance.head(10)
)

plt.show()

save_path = r"C:\Users\ABC\Downloads\cleaned_dataset.csv"

df.to_csv(save_path, index=False)

print(save_path)
print("Projected Completed Successfully")
