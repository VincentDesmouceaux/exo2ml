# %%
# -------------- Imports ---------------------
import polars as pl
import json
import os
import sys
import pandas as pd

sys.path.append("..")

from settings import (
    random_state,
    PROJECT_PATH,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
)

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


# %%
# -------------------- Data Loading ---------------------------
transactions = pl.read_parquet(
    os.path.join(PROJECT_PATH, "transactions_post_feature_engineering.parquet")
)

X = transactions.drop([REGRESSION_TARGET, CLASSIFICATION_TARGET])
y_regression = transactions[REGRESSION_TARGET]

with open("../features_used.json", "r") as f:
    feature_names = json.load(f)

with open("../categorical_features_used.json", "r") as f:
    categorical_features = json.load(f)

numerical_features = [col for col in feature_names if col not in categorical_features]

# %%
# ---------------------- Scaling -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X.to_pandas(), y_regression.to_pandas(), test_size=0.2, random_state=random_state
)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# %%
standard_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()

# %% Apply the scaling to the training set
X_train_scaled_1 = standard_scaler.fit_transform(X_train[numerical_features])
X_train_scaled_2 = min_max_scaler.fit_transform(X_train[numerical_features])


# %% Only use the transform method on the test set

X_test_scaled_1 = standard_scaler.transform(X_test[numerical_features])
X_test_scaled_2 = min_max_scaler.transform(X_test[numerical_features])
