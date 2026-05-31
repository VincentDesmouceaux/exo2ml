# %%
import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.append("..")
from sklearn.model_selection import (
    cross_validate,
    LeaveOneGroupOut,
)
import polars as pl
import pandas as pd
import json
import os
from settings import (
    random_state,
    PROJECT_PATH,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
)
from xgboost import XGBClassifier

# ------------------- Loading Data --------------------
# %%
transactions = pl.read_parquet(
    os.path.join(PROJECT_PATH, "transactions_post_feature_engineering.parquet")
)

with open("../features_used.json", "r") as f:
    feature_names = json.load(f)

with open("../categorical_features_used.json", "r") as f:
    categorical_features = json.load(f)


numerical_features = [col for col in feature_names if col not in categorical_features]

# ------------------------ Modeling --------------------------


# %%
def perform_cross_validation(
    X: pl.DataFrame,
    y: pl.Series,
    model,
    cross_val_type,
    scoring_metrics: tuple,
    return_estimator=False,
    groups=None,
):
    scores = cross_validate(
        model,
        X.to_numpy(),
        y.to_numpy(),
        cv=cross_val_type,
        return_train_score=True,
        return_estimator=return_estimator,
        scoring=scoring_metrics,
        groups=groups,
    )

    for metric in scoring_metrics:
        print(
            "Average Train {metric} : {metric_value}".format(
                metric=metric,
                metric_value=np.mean(scores["train_" + metric]),
            )
        )
        print(
            "Train {metric} Standard Deviation : {metric_value}".format(
                metric=metric, metric_value=np.std(scores["train_" + metric])
            )
        )

        print(
            "Average Test {metric} : {metric_value}".format(
                metric=metric, metric_value=np.mean(scores["test_" + metric])
            )
        )
        print(
            "Test {metric} Standard Deviation : {metric_value}".format(
                metric=metric, metric_value=np.std(scores["test_" + metric])
            )
        )

    return scores


# %%
xgboost_classifier = XGBClassifier(random_state=random_state)
classification_scoring_metrics = ("precision", "recall", "roc_auc")

# %%

# On va de nouveau travailler avec toutes les regions
X = transactions.drop([REGRESSION_TARGET, CLASSIFICATION_TARGET])
y_classification = transactions[CLASSIFICATION_TARGET]

# %%
X_group = X.to_pandas()

# %%
X_group["nom_region"] = pd.from_dummies(
    X_group[[col for col in X_group.columns if col.startswith("nom_region")]]
)

# %%
X_group = X_group.drop(
    [col for col in X_group.columns if col.startswith("nom_region_")], axis=1
)

# %%

scores_xgboost = perform_cross_validation(
    X=X_group[[col for col in feature_names if col in X_group.columns]],
    y=y_classification,
    model=xgboost_classifier,
    cross_val_type=LeaveOneGroupOut(),
    scoring_metrics=classification_scoring_metrics,
    groups=X_group["nom_region"].values,
)

# %%
scores_xgboost

# %%
