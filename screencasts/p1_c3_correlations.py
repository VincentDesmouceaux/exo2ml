# %%
import polars as pl
import json
import os
import sys

sys.path.append("..")

from settings import (
    PROJECT_PATH,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
)
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# ---------------------- Data Loading ---------------------------
transactions = pl.read_parquet(
    os.path.join(PROJECT_PATH, "transactions_post_feature_engineering.parquet")
)

X = transactions.drop([REGRESSION_TARGET, CLASSIFICATION_TARGET])
y_regression = transactions[REGRESSION_TARGET]
y_classification = transactions[CLASSIFICATION_TARGET]

# %%
feature_names = [col for col in X.columns if col != "id_transaction"]

# %%
categorical_features = [col for col in feature_names if "type_batiment" in col]
categorical_features.extend(["vefa", "ville_demandee"])
categorical_features.extend([col for col in feature_names if "nom_region" in col])

# %%
numerical_features = [col for col in feature_names if col not in categorical_features]

# %%
correlation_matrix = X.to_pandas()[numerical_features].corr()
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

"""
Le prix moyen au m2 du mois précédent est redodant avec le prix glissant à 6 mois, ce qui indique dans la majorité des cas une évolution lente du prix au m2 du marché
Pareil pour le nombre de transactions
L'annee est très corrélée à plusieurs informations 
Fraction depot banque très corrélé a
"""

# %%
highly_correlated_features = [
    "type_batiment_Maison",  # On garde l'une ou l'autre
    "n_pieces",  # On garde la surface habitable
    # "annee_transaction", # On garde le taux d'endettement
    "euros_par_habitant",  # Très corrélé avec le taux d'endettement
    "fraction_assurance_vie",
    "fraction_fonds_communs",
    "fraction_titres_non_action",
    "fraction_actions",
    "taux_endettement",
    "fraction_fond_pension",
    "variation_taux_endettement",
    "acceleration_taux_endettement",
    "indice_reference_loyers",
    "prix_m2_moyen_glissant_6mo",
    "nb_transaction_moyen_glissant_6mo",
    "fraction_depot_banque",
    "moyenne_glissante_6mo_variation_taux_interet",
]
# %%
feature_names = [col for col in feature_names if col not in highly_correlated_features]
feature_names = [
    col for col in feature_names if col not in ["prix_m2_moyen", "prix_m2_nombre"]
]

# %%
feature_names

# %%
numerical_features = [col for col in feature_names if col not in categorical_features]

# %%
import json

with open("features_used.json", "w") as f:
    json.dump(feature_names, f)

with open("categorical_features_used.json", "w") as f:
    json.dump(categorical_features, f)

# %%
X = X[feature_names]
