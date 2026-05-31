# %%
# -------------------------- Imports --------------------------
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
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
from catboost import CatBoostClassifier
from sklearn.svm import SVC

# %%
# -------------------------- Load Data --------------------------
transactions = pd.read_parquet(
    os.path.join(PROJECT_PATH, "transactions_post_feature_engineering.parquet")
)

transactions_extra_info = pd.read_parquet("../transactions_extra_infos.parquet")

# %%
with open("features_used.json", "r") as f:
    feature_names = json.load(f)

with open("categorical_features_used.json", "r") as f:
    categorical_features = json.load(f)

numerical_features = [col for col in feature_names if col not in categorical_features]


# %%
transactions = transactions.merge(
    transactions_extra_info[["id_transaction", "date_transaction"]],
    on="id_transaction",
    how="left",
)

transactions["anne_mois"] = transactions["date_transaction"].dt.to_period("M")

# %%
# -------------------------- Target Evolution Analysis --------------------------
class_proportion_evolution = transactions.groupby(["anne_mois"], as_index=False)[
    CLASSIFICATION_TARGET
].value_counts()

class_zero_evolution = class_proportion_evolution[
    class_proportion_evolution[CLASSIFICATION_TARGET] == 0
]

class_one_evolution = class_proportion_evolution[
    class_proportion_evolution[CLASSIFICATION_TARGET] == 1
]

plt.figure(figsize=(13, 10))
plt.bar(
    class_zero_evolution["anne_mois"].astype(str),
    class_zero_evolution["count"],
    color="red",
)
plt.bar(
    class_one_evolution["anne_mois"].astype(str),
    class_one_evolution["count"],
    color="green",
)
plt.xticks(rotation=90)
plt.legend(["Classe 0", "Classe 1"])

# %%
# -------------------------- Seperating Dataset into 3 periods --------------------------
transactions_pre_covid = transactions[transactions["anne_mois"] < "2020-03"]

transactions_covid = transactions[
    transactions["anne_mois"].between("2020-03", "2021-07")
]
transactions_post_covid = transactions[transactions["anne_mois"] > "2021-07"]

# %%
feature_names
# %%
# -------------------------- Data Example 1 --------------------------
feature_tested = "nb_transactions_mois_precedent"
sns.displot(
    transactions_pre_covid,
    x=feature_tested,
    kind="kde",
    fill=True,
)
sns.displot(
    transactions_covid,
    x=feature_tested,
    kind="kde",
    fill=True,
)

sns.displot(
    transactions_post_covid,
    x=feature_tested,
    kind="kde",
    fill=True,
)

# %%
# -------------------------- Data Example 2 --------------------------
feature_tested = "variation_taux_interet"
sns.displot(
    transactions_pre_covid,
    x=feature_tested,
    kind="kde",
    fill=True,
)
sns.displot(
    transactions_covid,
    x=feature_tested,
    kind="kde",
    fill=True,
)

sns.displot(
    transactions_post_covid,
    x=feature_tested,
    kind="kde",
    fill=True,
)

# %%
# -------------------------- Model Training on Pre-covid period --------------------------
from xgboost import XGBClassifier

xgboost_model = XGBClassifier(random_state=random_state)

xgboost_model.fit(
    transactions_pre_covid[feature_names],
    transactions_pre_covid[CLASSIFICATION_TARGET],
)

# %%
# -------------------------- Model Inference --------------------------
transactions_pre_covid["prediction"] = xgboost_model.predict(
    transactions_pre_covid[feature_names]
)
transactions_covid["prediction"] = xgboost_model.predict(
    transactions_covid[feature_names]
)
transactions_post_covid["prediction"] = xgboost_model.predict(
    transactions_post_covid[feature_names]
)

# %%
# -------------------------- Evidently Data Drift Analysis --------------------------
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

# %%
# ------------- Data Preprocessing -------------
column_mapping = ColumnMapping()

column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features
column_mapping.target = CLASSIFICATION_TARGET
column_mapping.prediction = "prediction"
# %%
# ------------- Drift Analysis -------------
data_drift_report = Report(metrics=[DataDriftPreset()])

data_drift_report.run(
    reference_data=transactions_pre_covid[feature_names],
    current_data=transactions_covid[feature_names],
    column_mapping=column_mapping,
)

# %%
data_drift_report

# %%

target_drift_report = Report(metrics=[TargetDriftPreset()])

target_drift_report.run(
    reference_data=transactions_pre_covid,
    current_data=transactions_covid,
    column_mapping=column_mapping,
)

# %%
target_drift_report
# %%
