import polars as pl
import numpy as np
from datetime import datetime
from settings import (
    TRANSACTIONS_FILE_PATH,
    TRANSACTION_DATE,
    TRANSACTION_MONTH,
    TRANSACTION_YEAR,
    DEPARTEMENT,
    CITY_UNIQUE_ID,
    SURFACE,
    AVERAGE_PRICE_PER_SQUARE_METER,
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET,
    PRICE_PER_SQUARE_METER,
    NB_TRANSACTIONS_PER_MONTH,
    REGION,
)
import seaborn as sns

# ---------------- Loading and basic processing functions -------------


def load_transactions(file_path: str = TRANSACTIONS_FILE_PATH) -> pl.DataFrame:
    arrays = dict(np.load(file_path))
    data = {
        k: (
            [s.decode("utf-8") for s in v.tobytes().split(b"\x00")]
            if v.dtype == np.uint8
            else v
        )
        for k, v in arrays.items()
    }
    transactions = pl.DataFrame(data)

    return transactions


def create_prix_au_m2_column(
    transactions: pl.DataFrame, price_col: str, square_meter_col: str
) -> pl.DataFrame:
    transactions = transactions.with_columns(
        (pl.col(price_col) / pl.col(square_meter_col)).alias(PRICE_PER_SQUARE_METER)
    )
    return transactions


def process_transactions(
    transactions: pl.DataFrame, lower_bound_date: datetime = datetime(2018, 1, 1)
) -> pl.DataFrame:
    filtered_transactions = transactions.filter(
        transactions[TRANSACTION_DATE] >= lower_bound_date
    )

    filtered_transactions = filtered_transactions.filter(
        pl.col("surface_terrains_nature") == "{}",
        pl.col("surface_terrains_sols") == "{}",
        pl.col("surface_terrains_agricoles") == "{}",
        pl.col("surface_locaux_industriels") == "{}",
        pl.col("surface_dependances") == "{}",
    )

    filtered_transactions = filtered_transactions.drop(
        [
            "surface_terrains_nature",
            "surface_terrains_sols",
            "surface_terrains_agricoles",
            "surface_locaux_industriels",
            "surface_dependances",
        ]
    )

    filtered_transactions = filtered_transactions.with_columns(
        pl.col(TRANSACTION_DATE).dt.month().alias(TRANSACTION_MONTH)
    )
    filtered_transactions = filtered_transactions.with_columns(
        pl.col(TRANSACTION_DATE).dt.year().alias(TRANSACTION_YEAR)
    )

    filtered_transactions = create_prix_au_m2_column(
        filtered_transactions, REGRESSION_TARGET, SURFACE
    )

    return filtered_transactions


def load_and_process_transactions(
    file_path: str = "transactions.npz",
    lower_bound_date: datetime = datetime(2018, 1, 1),
) -> pl.DataFrame:
    transactions = load_transactions(file_path=file_path)
    transactions = process_transactions(
        transactions=transactions, lower_bound_date=lower_bound_date
    )

    return transactions


def get_info_per_month_cities_enough_transactions(
    filtered_transactions: pl.DataFrame,
    grouping_cols=[
        "departement",
        "ville",
        "id_ville",
        TRANSACTION_YEAR,
        TRANSACTION_MONTH,
    ],
    threshold_nb_transactions=4,
    verbose=False,
):
    average_per_month_per_city = filtered_transactions.group_by(grouping_cols).agg(
        pl.col(PRICE_PER_SQUARE_METER).mean().name.suffix("_moyen"),
        pl.col(PRICE_PER_SQUARE_METER).count().alias(NB_TRANSACTIONS_PER_MONTH),
    )

    average_per_month_per_city_enough_transactions = average_per_month_per_city.filter(
        pl.col(NB_TRANSACTIONS_PER_MONTH) > threshold_nb_transactions
    )

    if verbose:
        average_per_month_per_city.select(
            pl.col(PRICE_PER_SQUARE_METER + "_moyen"),
            pl.col(NB_TRANSACTIONS_PER_MONTH),
        ).describe()

        # These are the cities that at least have 5 transactions per month
        cities_enough_transactions = (
            average_per_month_per_city_enough_transactions.group_by(CITY_UNIQUE_ID).agg(
                pl.col(NB_TRANSACTIONS_PER_MONTH).min().name.suffix("_nombre_min")
            )
        )

        100 * len(cities_enough_transactions) / len(average_per_month_per_city)
    else:
        pass

    return average_per_month_per_city_enough_transactions, average_per_month_per_city


def load_annual_macro_eco_context_data(
    taux_endettement_file_path: str,
    actifs_financiers_file_path: str,
):
    taux_endettement = pl.read_csv(taux_endettement_file_path)
    actifs_financiers = pl.read_csv(actifs_financiers_file_path)
    contexte_macro_eco_annuel = taux_endettement.join(
        actifs_financiers, on="date"
    )  # Le 2eme jeu de données remonte jusqu'aux années 90, on n'en n'a pas besoin

    return contexte_macro_eco_annuel


def load_regions_data(regions_file_path: str, departments_to_keep: list):
    departements_regions = pl.read_csv(regions_file_path)

    departements_regions = departements_regions.filter(
        pl.col("code_departement").is_in(departments_to_keep)
    ).with_columns(
        pl.col("code_departement").cast(pl.Int32).alias("departement"),
        pl.col("code_region").cast(pl.Int32).alias("region"),
    )

    return departements_regions


# ----------------- Datasets to be joined ----------------------------


# Tous ces datasets incluent des infos macro-economiques qui serviront en tant que features


def load_foyers_fiscaux(
    filepath: str,
    permietre_de_villes: pl.DataFrame,
    cols_to_keep: list = [
        "date",
        "departement",
        "id_ville",
        "ville",
        "n_foyers_fiscaux",
        "revenu_fiscal_moyen",
        "montant_impot_moyen",
    ],
) -> pl.DataFrame:
    foyers_fiscaux = pl.read_csv(filepath, infer_schema_length=None)

    foyers_fiscaux = foyers_fiscaux.filter(
        pl.col(DEPARTEMENT).is_in(
            [str(e) for e in permietre_de_villes[DEPARTEMENT].unique()]
        )
    ).with_columns([pl.col(e).cast(pl.Int32) for e in ["departement", "id_ville"]])

    foyers_fiscaux = foyers_fiscaux.join(
        permietre_de_villes, how="inner", on=permietre_de_villes.columns
    )

    foyers_fiscaux = foyers_fiscaux.select(cols_to_keep)

    return foyers_fiscaux


def load_monthly_macro_eco_context_data(
    taux_interet_path: str = TRANSACTIONS_FILE_PATH,
    nouveaux_emprunts_path: str = TRANSACTIONS_FILE_PATH,
    references_loyers_path: str = TRANSACTIONS_FILE_PATH,
):
    taux_interet = pl.read_csv(taux_interet_path, try_parse_dates=True)
    nouveaux_emprunts = pl.read_csv(nouveaux_emprunts_path, try_parse_dates=True)

    contexte_macro_eco_mensuel = taux_interet.join(nouveaux_emprunts, on="date")

    indices_reference_loyers = pl.read_csv(references_loyers_path, try_parse_dates=True)
    indices_reference_loyers = indices_reference_loyers.with_columns(
        pl.col("date").dt.year().alias("annee"), pl.col("date").dt.month().alias("mois")
    )

    contexte_macro_eco_mensuel = contexte_macro_eco_mensuel.with_columns(
        pl.col("date").dt.year().alias("annee"), pl.col("date").dt.month().alias("mois")
    )

    # Foward Fill car la donnée est trimestrielle et non mensuelle
    contexte_macro_eco_mensuel = (
        (
            contexte_macro_eco_mensuel.join(
                indices_reference_loyers, on=["annee", "mois"], how="left"
            )
            .sort(["annee", "mois"])
            .with_columns(pl.col("mois").forward_fill(), pl.col("IRL").forward_fill())
        )
        .drop("date_right")
        .rename({"taux": "taux_interet", "IRL": "indice_reference_loyers"})
    )

    return contexte_macro_eco_mensuel


def add_economical_context_features(
    transactions: pl.DataFrame,
    contexte_macro_eco_annuel: pl.DataFrame,
    contexte_macro_eco_mensuel: pl.DataFrame,
) -> pl.DataFrame:
    transactions_merged = transactions.join(
        contexte_macro_eco_annuel,
        left_on=TRANSACTION_YEAR,
        right_on="date",
        how="left",
    )

    transactions_merged = transactions_merged.join(
        contexte_macro_eco_mensuel,
        left_on=[TRANSACTION_YEAR, TRANSACTION_MONTH],
        right_on=["annee", "mois"],
        how="left",
    )

    return transactions_merged


def remove_departments_with_few_transactions(
    transactions: pl.DataFrame, threshold_percentile: float = 0.25, verbose: bool = True
) -> pl.DataFrame:
    transactions_per_department = (
        transactions.select(DEPARTEMENT).to_series().value_counts()
    )

    if verbose:
        print(transactions_per_department.describe())
        sns.displot(transactions_per_department)
    else:
        pass

    threshold = (
        transactions_per_department.quantile(threshold_percentile)
        .select("count")
        .to_series()
        .to_list()[0]
    )

    departments_to_keep = (
        transactions_per_department.filter(pl.col("count") > threshold)
        .select(DEPARTEMENT)
        .to_series()
        .to_list()
    )

    transactions_filtered = transactions.filter(
        pl.col(DEPARTEMENT).is_in(departments_to_keep)
    )

    return transactions_filtered, departments_to_keep


def remove_regions_with_few_transactions(
    filtered_transactions: pl.DataFrame,
    nb_regions_to_keep: int = 5,
):
    regions_avec_plus_de_transactions = (
        filtered_transactions.select(REGION)
        .to_series()
        .value_counts()
        .sort(by="count", descending=True)
        .head(nb_regions_to_keep)
        .select(REGION)
        .to_series()
        .to_list()
    )

    filtered_transactions = filtered_transactions.filter(
        pl.col(REGION).is_in(regions_avec_plus_de_transactions)
    )

    return filtered_transactions


# ----------------- Classification Target Calculation ----------------------------


def add_classification_target_to_transactions(
    filtered_transactions: pl.DataFrame,
    target_col_name: str = CLASSIFICATION_TARGET,
    percentage_below_mean: float = 0.1,
) -> pl.DataFrame:
    filtered_transactions = filtered_transactions.with_columns(
        pl.when(
            pl.col(PRICE_PER_SQUARE_METER)
            < (
                pl.col(AVERAGE_PRICE_PER_SQUARE_METER)
                - (percentage_below_mean * pl.col(AVERAGE_PRICE_PER_SQUARE_METER))
            )
        )
        .then(1)
        .otherwise(0)
        .alias(target_col_name)
    )
    return filtered_transactions
