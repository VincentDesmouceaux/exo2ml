# %%
# -------------------------- Imports --------------------------
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import polars as pl
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
from catboost import CatBoostClassifier, Pool
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
import mlflow
from tqdm import tqdm

# %%


def perform_cross_validation(
    X: pl.DataFrame,
    y: pl.Series,
    model,
    cross_val_type,
    scoring_metrics: tuple,
    groups=None,
):
    scores = cross_validate(
        model,
        X.to_numpy(),
        y.to_numpy(),
        cv=cross_val_type,
        return_train_score=True,
        return_estimator=True,
        scoring=scoring_metrics,
        groups=groups,
    )

    scores_dict = {}
    for metric in scoring_metrics:
        scores_dict["average_train_" + metric] = np.mean(scores["train_" + metric])
        scores_dict["train_" + metric + "_std"] = np.std(scores["train_" + metric])
        scores_dict["average_test_" + metric] = np.mean(scores["test_" + metric])
        scores_dict["test_" + metric + "_std"] = np.std(scores["test_" + metric])

    model.fit(X.to_numpy(), y.to_numpy())

    return scores, scores_dict, model


# %%
def get_features_most_importance(importances, feature_names, threshold=0.8):
    sorted_indices = np.argsort(importances)
    sorted_importances = importances[sorted_indices][::-1]
    sorted_feature_names = [feature_names[i] for i in sorted_indices][::-1]

    cumulated_importance = 0
    important_features = []

    for importance, feature_name in zip(sorted_importances, sorted_feature_names):
        cumulated_importance += importance
        important_features.append(feature_name)

        if cumulated_importance >= threshold:
            break

    return important_features


# %%
# -------------------------- Load Data --------------------------
transactions = pl.read_parquet(
    os.path.join(PROJECT_PATH, "transactions_post_feature_engineering.parquet")
)


with open("features_used.json", "r") as f:
    feature_names = json.load(f)

with open("categorical_features_used.json", "r") as f:
    categorical_features = json.load(f)

numerical_features = [col for col in feature_names if col not in categorical_features]

# %%
feature_names
# %%
# -------------------------- Data Split for screencast & exercice purposes --------------------------

transactions_v1 = transactions.filter(pl.col("annee_transaction") < 2020)

transactions_v2 = transactions.filter(
    pl.col("annee_transaction").is_between(2020, 2021)
)
features_1 = [
    "type_batiment_Appartement",
    "surface_habitable",
    "prix_m2_moyen_mois_precedent",
    "nb_transactions_mois_precedent",
    "taux_interet",
    "variation_taux_interet",
    "acceleration_taux_interet",
]

features_2 = features_1.extend(["longitude", "latitude", "vefa"])


# %%
from mlflow import MlflowClient

client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

# %% 
mlflow.search_runs(search_all_experiments=True)

# %%
# ------------------------ Premiere modèlisation ----------------


for region in tqdm(
    [
        "Île-de-France",
        "Auvergne-Rhône-Alpes",
        "Nouvelle-Aquitaine",
    ]
):
    region_transactions_v1 = transactions_v1.filter(pl.col("nom_region_" + region) == 1)

    experiment_tags = {
        "region": region,
        "revision_de_donnees": "v1",
        "date_de_construction": "Fin 2019",
    }

    try:
        region_experiment = client.create_experiment(region, tags=experiment_tags)
    except:
        region_experiment = mlflow.set_experiment(region)

    # ------------------------ 1er Run Modèle Dummy  ----------------
    with mlflow.start_run(run_name="dummy_run_" + region) as run:
        print(
            " ------------------Running Dummy Model for Region: ",
            region + "------------------",
        )
        X = region_transactions_v1.drop(
            [REGRESSION_TARGET, CLASSIFICATION_TARGET]
        ).to_pandas()
        y_classification = region_transactions_v1[CLASSIFICATION_TARGET].to_pandas()

        dummy_classifier = DummyClassifier(strategy="most_frequent")
        classification_scoring_metrics = ["recall", "precision", "f1"]

        scores, scores_dict, dummy_classifier = perform_cross_validation(
            X=X[features_1],
            y=y_classification,
            model=dummy_classifier,
            cross_val_type=StratifiedKFold(),
            scoring_metrics=classification_scoring_metrics,
        )

        mlflow.log_param("random_state", random_state)
        mlflow.log_param("features", features_1)

        mlflow.log_metrics(scores_dict)

        mlflow.sklearn.log_model(dummy_classifier, "dummy_classifier")

        dataset_abstraction = mlflow.data.from_pandas(
            region_transactions_v1.to_pandas()
        )
        mlflow.log_input(dataset_abstraction)

    # ------------------------ 2eme Run Modèle Catboost  ----------------
    with mlflow.start_run(run_name="catboost_" + region) as run:
        print(
            " ------------------Running Catboost Model for Region: ",
            region + "------------------",
        )
        X = region_transactions_v1.drop(
            [REGRESSION_TARGET, CLASSIFICATION_TARGET]
        ).to_pandas()
        y_classification = region_transactions_v1[CLASSIFICATION_TARGET].to_pandas()

        catboost_model = CatBoostClassifier(random_state=random_state, verbose=False)
        classification_scoring_metrics = ["recall", "precision", "f1"]

        scores, scores_dict, catboost_model = perform_cross_validation(
            X=X[features_1],
            y=y_classification,
            model=catboost_model,
            cross_val_type=StratifiedKFold(),
            scoring_metrics=classification_scoring_metrics,
        )

        mlflow.log_param("random_state", random_state)
        mlflow.log_param("features", features_1)

        for metric, value in scores_dict.items():
            mlflow.log_metric(metric, value)

        mlflow.sklearn.log_model(catboost_model, "catboost_classifier")

        dataset_abstraction = mlflow.data.from_pandas(
            region_transactions_v1.to_pandas()
        )
        mlflow.log_input(dataset_abstraction)

    feature_importances = catboost_model.get_feature_importance(Pool(X[features_1]))
    most_important_features = get_features_most_importance(
        feature_importances, features_1
    )

    # ------------------------ 3eme Run Modèle Catboost Leger  ----------------
    with mlflow.start_run(run_name="catboost_light_" + region) as run:
        print(
            " ------------------Running Catboost Light Model for Region: ",
            region + "------------------",
        )
        X = region_transactions_v1.drop([REGRESSION_TARGET, CLASSIFICATION_TARGET])
        y_classification = region_transactions_v1[CLASSIFICATION_TARGET]

        catboost_light_model = CatBoostClassifier(
            random_state=random_state, verbose=False
        )
        classification_scoring_metrics = ["recall", "precision", "f1"]

        scores, scores_dict, catboost_light_model = perform_cross_validation(
            X=X[most_important_features],
            y=y_classification,
            model=catboost_light_model,
            cross_val_type=StratifiedKFold(),
            scoring_metrics=classification_scoring_metrics,
        )

        mlflow.log_param("random_state", random_state)
        mlflow.log_param("features", most_important_features)

        for metric, value in scores_dict.items():
            mlflow.log_metric(metric, value)

        mlflow.sklearn.log_model(catboost_light_model, "catboost_light_classifier")

        dataset_abstraction = mlflow.data.from_pandas(
            region_transactions_v1.to_pandas()
        )
        mlflow.log_input(dataset_abstraction)


# %%
# ------------------------ Deuxième de modélisation ----------------

nouvelle_acquitaine_experiment = client.set_experiment("Nouvelle-Aquitaine")

client = mlflow.tracking.MlflowClient()



with mlflow.start_run(run_name="catboost_" + region) as run:
        print(
            " ------------------Running Catboost Model for Region: ",
            region + "------------------",
        )
        X = region_transactions_v1.drop(
            [REGRESSION_TARGET, CLASSIFICATION_TARGET]
        ).to_pandas()
        y_classification = region_transactions_v1[CLASSIFICATION_TARGET].to_pandas()

        catboost_model = CatBoostClassifier(random_state=random_state, verbose=False)
        classification_scoring_metrics = ["recall", "precision", "f1"]

        scores, scores_dict, catboost_model = perform_cross_validation(
            X=X[features_1],
            y=y_classification,
            model=catboost_model,
            cross_val_type=StratifiedKFold(),
            scoring_metrics=classification_scoring_metrics,
        )

        mlflow.log_param("random_state", random_state)
        mlflow.log_param("features", features_1)

        for metric, value in scores_dict.items():
            mlflow.log_metric(metric, value)

        mlflow.sklearn.log_model(catboost_model, "catboost_classifier")

        dataset_abstraction = mlflow.data.from_pandas(
            region_transactions_v1.to_pandas()
        )
        mlflow.log_input(dataset_abstraction)

    feature_importances = catboost_model.get_feature_importance(Pool(X[features_1]))
    most_important_features = get_features_most_importance(
        feature_importances, features_1
    )

