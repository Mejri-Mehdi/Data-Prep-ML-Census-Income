import os
import pandas as pd
import numpy as np

columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_path = os.path.join(base_dir, 'data', 'raw', 'adult.data')
test_path = os.path.join(base_dir, 'data', 'raw', 'adult.test')

df_train = pd.read_csv(train_path, header=None, names=columns, skipinitialspace=True)
df_test = pd.read_csv(test_path, header=None, names=columns, skipinitialspace=True, skiprows=1)

print(f"=== DATASET SHAPES ===")
print(f"Train Shape: {df_train.shape}")
print(f"Test Shape:  {df_test.shape}")
print(f"Total rows:  {len(df_train) + len(df_test)}")

print(f"\n=== TARGET VARIABLE DISTRIBUTION ===")
print("Train:")
print(df_train['income'].value_counts(normalize=True).round(4) * 100)
print("\nTest (Notice trailing periods in raw test):")
print(df_test['income'].value_counts(normalize=True).round(4) * 100)

print(f"\n=== MISSING VALUES ('?' representation) ===")
for col in df_train.columns:
    q_tr = (df_train[col] == '?').sum()
    q_te = (df_test[col] == '?').sum()
    if q_tr > 0 or q_te > 0:
        print(f"{col:15s} -> Train: {q_tr:5d} ({q_tr/len(df_train)*100:.2f}%) | Test: {q_te:5d} ({q_te/len(df_test)*100:.2f}%)")

print(f"\n=== DUPLICATE ROWS ===")
print(f"Train Duplicates: {df_train.duplicated().sum()}")
print(f"Test Duplicates:  {df_test.duplicated().sum()}")

print(f"\n=== NUMERICAL FEATURES SUMMARY (Train) ===")
print(df_train.describe().round(2).to_string())

print(f"\n=== CATEGORICAL FEATURES CARDINALITY ===")
for col in df_train.select_dtypes(include=['object']).columns:
    if col != 'income':
        print(f"{col:15s}: {df_train[col].nunique()} unique categories")
