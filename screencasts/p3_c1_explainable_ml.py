# %%
# -------------------------- Imports --------------------------s
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import polars as pl
import pandas as pd
import numpy as np
import json
import os
import sys
import seaborn as sns

sns.set()

sys.path.append("..")

from settings import (
    random_state,
    PROJECT_PATH,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %%
# -------------------------- Load Data --------------------------
transactions = pl.read_parquet(
    os.path.join(PROJECT_PATH, "transactions_post_feature_engineering.parquet")
)

selected_region = "nom_region_Occitanie"
region_transactions = transactions.filter(pl.col(selected_region) == 1)

X = region_transactions.drop([REGRESSION_TARGET, CLASSIFICATION_TARGET])
y_regression = region_transactions[REGRESSION_TARGET]
y_classification = region_transactions[CLASSIFICATION_TARGET]

with open("features_used.json", "r") as f:
    feature_names = json.load(f)

with open("categorical_features_used.json", "r") as f:
    categorical_features = json.load(f)

numerical_features = [col for col in feature_names if col not in categorical_features]

# %%
# -------------------------- Train Model --------------------------
rf_regressor = RandomForestRegressor(random_state=random_state)

X_train, X_test, y_train, y_test = train_test_split(
    X.to_pandas(), y_regression.to_pandas(), random_state=random_state
)

rf_regressor.fit(X_train[feature_names], y_train)


# %%
# ---------------------- Trees Feature Importance ----------------------
importances = rf_regressor.feature_importances_
indices = np.argsort(importances)
std = np.std([tree.feature_importances_ for tree in rf_regressor.estimators_], axis=0)

# %%

plt.figure(1)
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], align="center", xerr=std[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")

# %%
# ------------ Methode 1 ---------------
feature_names_simplified = [
    "surface_habitable",
    "prix_m2_moyen_mois_precedent",
    "longitude",
    "latitude",
    "nb_transactions_mois_precedent",
    "type_batiment_Appartement",
]


# %%
# ------------ Methode 2 ---------------
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
            print("Cumulated importance Reached: ", cumulated_importance)
            break

    return important_features


important_features = get_features_most_importance(importances, feature_names)
print(important_features)
# %%
# --------------------- Train Light Model ---------------------
rf_regressor_light = RandomForestRegressor(random_state=random_state)
rf_regressor_light.fit(X_train[feature_names_simplified], y_train)

# %%
# -------------------------- Performance metrics Full Model  --------------------------
y_train_pred = rf_regressor.predict(X_train[feature_names])
y_test_pred = rf_regressor.predict(X_test[feature_names])

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print("Train RMSE : ", train_rmse)
train_mae = mean_absolute_error(y_train, y_train_pred)
print("Train MAE : ", train_mae)
train_r2 = r2_score(y_train, y_train_pred)
print("Train R2 : ", train_r2)

test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print("Test RMSE : ", test_rmse)
test_mae = mean_absolute_error(y_test, y_test_pred)
print("Test MAE : ", test_mae)
test_r2 = r2_score(y_test, y_test_pred)
print("Test R2 : ", test_r2)

# %%
# -------------------------- Performance metrics Light Model  --------------------------
y_train_pred = rf_regressor_light.predict(X_train[feature_names_simplified])
y_test_pred = rf_regressor_light.predict(X_test[feature_names_simplified])

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print("Train RMSE : ", train_rmse)
train_mae = mean_absolute_error(y_train, y_train_pred)
print("Train MAE : ", train_mae)
train_r2 = r2_score(y_train, y_train_pred)
print("Train R2 : ", train_r2)

test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print("Test RMSE : ", test_rmse)
test_mae = mean_absolute_error(y_test, y_test_pred)
print("Test MAE : ", test_mae)
test_r2 = r2_score(y_test, y_test_pred)
print("Test R2 : ", test_r2)

# %%
# -------------------------- Another Method : Permutation Importance  --------------------------

from sklearn.inspection import permutation_importance

results = permutation_importance(
    rf_regressor, X_train[feature_names], y_train, scoring="r2"
)

# %%
permutation_importances = results.importances_mean
permutation_indices = np.argsort(importances)

# %%

plt.figure(1)
plt.title("Feature Importances")
plt.barh(
    range(len(permutation_indices)),
    permutation_importances[permutation_indices],
    align="center",
    xerr=std[permutation_indices],
)
plt.yticks(
    range(len(permutation_indices)), [feature_names[i] for i in permutation_indices]
)
plt.xlabel("Permutation Importances")


# %%
# --- Aller plus loin ------
results
# %%
# -------------------------- SHAP Usage --------------------------
import shap
from shap import (
    TreeExplainer,
    KernelExplainer,
    DeepExplainer,
    GradientExplainer,
    ExactExplainer,
)

# %%
explainer = TreeExplainer(rf_regressor_light, approximate=True)
shap_values_train = explainer(X_train[feature_names_simplified])

# %%
shap_values_test = explainer(X_test[feature_names_simplified])


# %%
shap.plots.waterfall(shap_values_train[0])

# %%
shap.plots.waterfall(shap_values_test[5])

# %%
shap.plots.beeswarm(shap_values_train)


# %%
