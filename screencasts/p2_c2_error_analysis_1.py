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
from catboost import CatBoostRegressor

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
# -------------------------- Train Model --------------------------

catboost_regressor = CatBoostRegressor(random_state=random_state, verbose=False)

X_train, X_test, y_train, y_test = train_test_split(
    X.to_pandas(), y_regression.to_pandas(), random_state=random_state
)

catboost_regressor.fit(X_train[feature_names], y_train)

# %%
## -------------------------- Error Analysis --------------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_train_pred = catboost_regressor.predict(X_train[feature_names])
y_test_pred = catboost_regressor.predict(X_test[feature_names])

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

# -------------------------- Detailled Error Analysis --------------------------


# %%
absolute_errors_train = pd.Series(
    [
        abs(true_value - predicted_value)
        for (true_value, predicted_value) in zip(y_train, y_train_pred)
    ]
)

absolute_errors_test = pd.Series(
    [
        abs(true_value - predicted_value)
        for (true_value, predicted_value) in zip(y_test, y_test_pred)
    ]
)

print(absolute_errors_train.describe().apply(lambda x: format(x, "f")))
print(absolute_errors_test.describe().apply(lambda x: format(x, "f")))


# %%
def plot_regression_predictions(
    model, X, y, title="Regression Model: Predicted vs Target"
):
    y_pred = model.predict(X)

    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot(
        [min(y), max(y)],
        [min(y), max(y)],
        color="red",
        linestyle="--",
    )
    plt.xlabel("Target")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.show()


# %%
def plot_regression_error(
    model,
    X,
    y,
    xlim_min=-1 * 10**5,
    xlim_max=10**5,
    title="Distribution of Regression Model Error",
):
    y_pred = model.predict(X)
    errors = y - y_pred

    sns.displot(errors)
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.xlim(xlim_min, xlim_max)
    plt.title(title)
    plt.show()


# %%
plot_regression_predictions(catboost_regressor, X_train[feature_names], y_train)

# %%
plot_regression_predictions(catboost_regressor, X_test[feature_names], y_test)

# %%
plot_regression_error(
    catboost_regressor,
    X_train[feature_names],
    y_train,
    xlim_min=-1 * 10**5,
    xlim_max=10**5,
)

# %%
plot_regression_error(
    catboost_regressor,
    X_test[feature_names],
    y_test,
    xlim_min=-1 * 10**5,
    xlim_max=10**5,
)

# -------------------------- Error Breakdown per Feature --------------------------


# %%


def get_X_y_particular_feature_value(
    X: pd.DataFrame, y: pd.Series, feature_col: str, feature_value
):
    X["target"] = y
    X_filtered = X[X[feature_col] == feature_value]

    return X_filtered.drop(["target"], axis=1), X_filtered["target"]


regions_features = [col for col in feature_names if col.startswith("nom_region")]
X_train_y_train_regions = {
    region: get_X_y_particular_feature_value(X_train, y_train, region, 1)
    for region in regions_features
}

X_test_y_test_regions = {
    region: get_X_y_particular_feature_value(X_test, y_test, region, 1)
    for region in regions_features
}

# %%
for region in regions_features:
    plt.figure(figsize=(13, 10))
    plot_regression_error(
        catboost_regressor,
        X_test_y_test_regions[region][0][feature_names],
        X_test_y_test_regions[region][1],
        title="Distribution of Regression Model Error - {region}".format(region=region),
    )

# %%
for region in regions_features:
    plot_regression_predictions(
        catboost_regressor,
        X_test_y_test_regions[region][0][feature_names],
        X_test_y_test_regions[region][1],
        title="Regression Model: Predicted vs Target - {region}".format(region=region),
    )

# %%
# --- Create Error Columns ---

y_test_pred = catboost_regressor.predict(X_test[feature_names])
errors_test = y_test - y_test_pred
X_test["error"] = errors_test

for feature in numerical_features:
    plt.figure(figsize=(13, 10))
    sns.scatterplot(x=feature, y="error", data=X_test)
    plt.title("Error Distribution per {feature}".format(feature=feature))
    plt.show()

# %%
