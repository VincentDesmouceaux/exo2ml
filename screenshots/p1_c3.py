import polars as pl
import os
from settings import PROJECT_PATH
import matplotlib.pyplot as plt
import seaborn as sns

transactions = pl.read_parquet(
    os.path.join(PROJECT_PATH, "transactions_post_feature_engineering.parquet")
)

feature_names = [col for col in transactions.columns if col != "id_transaction"]

# %%
categorical_features = [col for col in feature_names if "type_batiment" in col]
categorical_features.append("vefa")
categorical_features.extend([col for col in feature_names if "nom_region" in col])

# %%
numerical_features = [col for col in feature_names if col not in categorical_features]
numerical_features.remove("nom_departement")
# %%
correlation_matrix = transactions.to_pandas()[numerical_features].corr()
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


highly_correlated_features = [
    "type_batiment_Maison",  # On garde l'une ou l'autre
    "n_pieces",  # On garde la surface habitable
    # "annee_transaction", # On garde le taux d'endettement
    "euros_par_habitant",  # Très corrélé avec le taux d'endettement
    "montant_impot_moyen",  # Très corrélé avec le revenu fiscal moyen
    "fraction_assurance_vie",
    "fraction_fonds_communs",
    "fraction_titres_non_action",
    "fraction_actions",
    "taux",
    "IRL",
]
