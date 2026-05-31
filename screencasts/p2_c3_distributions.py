# %%
# -------------------- Imports --------------------
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import polars as pl
import pandas as pd
import json
import os
import sys

sys.path.append("..")

import seaborn as sns

sns.set()
from settings import (
    random_state,
    PROJECT_PATH,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
)
from sklearn.ensemble import RandomForestClassifier

# %%
# -------------------------- Load Data --------------------------
transactions = pl.read_parquet(
    os.path.join(PROJECT_PATH, "transactions_post_feature_engineering.parquet")
)

X = transactions.drop([REGRESSION_TARGET, CLASSIFICATION_TARGET])
y_regression = transactions[REGRESSION_TARGET]
y_classification = transactions[CLASSIFICATION_TARGET]


with open("features_used.json", "r") as f:
    feature_names = json.load(f)

with open("categorical_features_used.json", "r") as f:
    categorical_features = json.load(f)


numerical_features = [col for col in feature_names if col not in categorical_features]

# %%
# ------------------ Fitting Model ------------------
classifier = RandomForestClassifier(random_state=random_state, verbose=False)

X_train, X_test, y_train, y_test = train_test_split(
    X.to_pandas(), y_classification.to_pandas(), random_state=random_state
)

classifier.fit(X_train[feature_names], y_train)

# %%
# ------------------ Evaluating Model ------------------
from sklearn.metrics import confusion_matrix, classification_report

y_train_pred = classifier.predict(X_train[feature_names])
y_test_pred = classifier.predict(X_test[feature_names])


confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
print(confusion_matrix_train)

classification_report_train = classification_report(y_train, y_train_pred)
print(classification_report_train)

confusion_matrix_test = confusion_matrix(y_test, y_test_pred)
print(confusion_matrix_test)

classification_report_test = classification_report(y_test, y_test_pred)
print(classification_report_test)
# %%

# ------------------ Probability Score Analysis  ------------------
X_train["prediction"] = y_train_pred
X_train["probability_score"] = classifier.predict_proba(
    X_train[feature_names]
)[:, 1]

X_test["prediction"] = y_test_pred
X_test["probability_score"] = classifier.predict_proba(X_test[feature_names])[
    :, 1
]


# %%
def get_prediction_type(prediction, target):
    if prediction == 1 and target == 1:
        return "true_positive"
    elif prediction == 0 and target == 0:
        return "true_negative"
    elif prediction == 1 and target == 0:
        return "false_positive"
    elif prediction == 0 and target == 1:
        return "false_negative"
    else:
        return "unknown"


# %%
X_train[CLASSIFICATION_TARGET] = y_train
X_train["prediction_type"] = X_train.apply(
    lambda row: get_prediction_type(row["prediction"], row[CLASSIFICATION_TARGET]),
    axis=1,
)

X_test[CLASSIFICATION_TARGET] = y_test
X_test["prediction_type"] = X_test.apply(
    lambda row: get_prediction_type(row["prediction"], row[CLASSIFICATION_TARGET]),
    axis=1,
)

# %%

color_palette = {
    "true_positive": "green", 
    "false_positive": "red",
    "false_negative": "orange",
    "true_negative": "blue"
    }
def plot_probability_distribution_per_prediction_type(
        X, 
        color_palette,
        categories_to_excelude=["true_negative"]
    ):
    X_filtered = X[~X["prediction_type"].isin(categories_to_excelude)]
    sns.displot(data=X_filtered, x="probability_score", hue="prediction_type", palette=color_palette)
    plt.legend(loc="upper right")
    plt.show()


# %%
plot_probability_distribution_per_prediction_type(
    X_test, 
    color_palette, 
    categories_to_excelude=["true_negative"]
)
# %%
plot_probability_distribution_per_prediction_type(
    X_test, 
    color_palette, 
    categories_to_excelude=["true_negative", "false_negative"]
)

# %%
plot_probability_distribution_per_prediction_type(
    X_test, 
    color_palette, 
    categories_to_excelude=["true_negative", "false_positive"]
)
# %%
