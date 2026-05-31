# %%

import polars as pl
import json
import os
import sys

sys.path.append("..")

from settings import (
    random_state,
    PROJECT_PATH,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
)

from xgboost import XGBClassifier, XGBRegressor

# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import bentoml

# %%

transactions = pl.read_parquet(
    os.path.join(PROJECT_PATH, "transactions_post_feature_engineering.parquet")
)

# %%

X = transactions.drop([REGRESSION_TARGET, CLASSIFICATION_TARGET]).to_pandas()
y_regression = transactions[REGRESSION_TARGET].to_pandas()
y_classification = transactions[CLASSIFICATION_TARGET].to_pandas()

# %%
with open("../features_used.json", "r") as f:
    feature_names = json.load(f)

with open("../categorical_features_used.json", "r") as f:
    categorical_features = json.load(f)

# %%
numerical_features = [col for col in feature_names if col not in categorical_features]

# %%

xgb_regressor = XGBRegressor(random_state=random_state)
xgb_classifier = XGBClassifier(random_state=random_state)

# %%
xgb_regressor.fit(X[feature_names], y_regression)
xgb_classifier.fit(X[feature_names], y_classification)

# %%

bentoml.sklearn.save_model("transaction_value_estimator", xgb_regressor)
bentoml.sklearn.save_model("transaction_below_market_identifier", xgb_classifier)

# %%
feature_names
# %%
import numpy as np

test_values = np.array(
    [[list(X[feature].sample())[0] for feature in feature_names] for i in range(10)]
)


# %%
test_values
# %%
with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    result = client.predict(transactions=test_values)

# %%
result[0]

# %%
result[1]
