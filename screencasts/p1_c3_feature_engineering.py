# %%
import polars as pl
import os
import sys

sys.path.append("..")
from settings import (
    PROJECT_PATH,
    TAUX_ENDETTEMENT_FILE_PATH,
    ACTIFS_FINANCIERS_FILE_PATH,
    TAUX_DINTERET_FILE_PATH,
    FLUX_EMPRUNTS_FILE_PATH,
    INDICE_REFERENCE_LOYERS,
)

from data_processing_functions import (
    load_annual_macro_eco_context_data,
    load_monthly_macro_eco_context_data,
    add_economical_context_features,
)
from feature_engineering_functions import (
    compute_city_features,
    compute_price_per_m2_features,
    create_debt_ratio_features,
    calculate_interest_rate_features,
)

# %%


transactions = pl.read_parquet(
    os.path.join(PROJECT_PATH, "transactions_immobilieres.parquet")
)

transactions = transactions.with_columns(
    pl.col("departement").cast(pl.Int32),
    pl.col("mois_transaction").cast(pl.Int32),
    pl.col("vefa").cast(pl.Int32),
)

transactions_per_city = pl.read_parquet(
    os.path.join(PROJECT_PATH, "transactions_par_ville.parquet")
)

transactions_per_city = transactions_per_city.with_columns(
    pl.col("departement").cast(pl.Int32),
    pl.col("mois_transaction").cast(pl.Int32),
)
# %%
transactions.head()

# %%
# -------------- Création de features à partir d'un seuil sur une autre ---------------

# %%
transactions_per_city.head()

# %%

transactions_per_city = compute_city_features(
    transactions_per_city,
    feature_name="ville_demandee",
    quantile_threshold=0.8,
    verbose=True,
)


# %%
transactions_per_city["ville_demandee"].describe()

# %%

# ---------------- Création de features en utilisant du "lagging" --------------

transactions_per_city
# %%

transactions_per_city = compute_price_per_m2_features(
    transactions_per_city,
    sort_columns=[
        "departement",
        "ville",
        "id_ville",
        "annee_transaction",
        "mois_transaction",
    ],
)

# %%

transactions = transactions.join(
    transactions_per_city,
    on=[
        "departement",
        "ville",
        "id_ville",
        "annee_transaction",
        "mois_transaction",
        "prix_m2_moyen",
        "nb_transactions_mois",
    ],
    how="inner",
)

# %%
# ------------ Creation de features représentant des variations à court terme -------
annual_macro_eco_context = load_annual_macro_eco_context_data(
    TAUX_ENDETTEMENT_FILE_PATH, ACTIFS_FINANCIERS_FILE_PATH
)

# %%
annual_macro_eco_context
# %%

annual_macro_eco_context = create_debt_ratio_features(annual_macro_eco_context)


# %%
# ------------ Creation de features reorésentant des tendances à plus long terme  -------
monthly_macro_eco_context = load_monthly_macro_eco_context_data(
    TAUX_DINTERET_FILE_PATH, FLUX_EMPRUNTS_FILE_PATH, INDICE_REFERENCE_LOYERS
)

monthly_macro_eco_context = monthly_macro_eco_context.with_columns(
    pl.col("mois").cast(pl.Int32)
)

# %%
monthly_macro_eco_context
# %%

monthly_macro_eco_context = calculate_interest_rate_features(
    monthly_macro_eco_context,
    aggregation_period="6mo",
    interest_rate_col="taux_interet",
)


# %%

transactions = add_economical_context_features(
    transactions,
    annual_macro_eco_context,
    monthly_macro_eco_context,
)


# %%
# ----------------- One Hot Encoding -------------------------
transactions = transactions.to_dummies(columns=["type_batiment"])
transactions = transactions.to_dummies(columns=["nom_region"])

# %%
transactions

# %%
transactions.write_parquet(
    os.path.join(PROJECT_PATH, "transactions_post_feature_engineering.parquet")
)


# %%

"""
On construit un dataframe avec des infos "extra" séparées des transactions. 
Ces infos peuvent servir à l'interprétation.
"""

cols_extra_info = [
    "id_transaction",
    "date_transaction",
    "id_ville",
    "ville",
    "code_departement",
    "departement",
    "code_region",
    "region",
    "adresse",
    "code_postal",
    "id_parcelle_cadastre",
    "prix_m2",  # Pour eviter du Data Leakage
    "prix_m2_moyen",  # Pour eviter du Data Leakage
    "nom_departement",  # A exclure pour le moment
]

transactions_extra_infos = transactions.select(
    [e for e in cols_extra_info if e in transactions.columns]
)

cols_to_drop = [
    "USD_par_habitant",
    "date",
    "ville_right",
]
cols_to_drop.extend(cols_extra_info)
cols_to_drop.remove("id_transaction")

transactions = transactions.drop(
    [col for col in transactions.columns if col in cols_to_drop]
)


# %%
transactions
# %%

transactions = transactions.filter(
    pl.all_horizontal(pl.col(pl.Float32, pl.Float64, pl.Int32, pl.Int64).is_not_nan())
)

# %%


# %%
