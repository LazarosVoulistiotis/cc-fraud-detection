import pandas as pd
from eval_harness import stratified_train_valid_test_split
from pathlib import Path

DATA_PATH = Path("data/data_raw/creditcard.csv")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Class"])
y = df["Class"].values

X_train, X_valid, X_test, y_train, y_valid, y_test = stratified_train_valid_test_split(X, y)
print(y_train.mean(), y_valid.mean(), y_test.mean())
