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
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve

# %%
# -------------------------- Load Data --------------------------
transactions = pl.read_parquet(
    os.path.join(PROJECT_PATH, "transactions_post_feature_engineering.parquet")
)

X = transactions.drop([REGRESSION_TARGET, CLASSIFICATION_TARGET])
y_regression = transactions[REGRESSION_TARGET]
y_classification = transactions[CLASSIFICATION_TARGET]


with open("../features_used.json", "r") as f:
    feature_names = json.load(f)

with open("../categorical_features_used.json", "r") as f:
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

y_scores_test = classifier.predict_proba(X_test[feature_names])[:, 1]

precision_test, recall_test, thresholds_test = precision_recall_curve(
    y_test, y_scores_test
)

plt.plot(recall_test, precision_test, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Test Set")
plt.legend()
plt.show()

auc_test = auc(recall_test, precision_test)

# %%

auc_test
# %%

# %%
# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="ROC Curve (AUC = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

# %%
