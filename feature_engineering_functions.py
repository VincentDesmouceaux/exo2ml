import polars as pl
from settings import (
    NB_TRANSACTIONS_PER_MONTH,
    DEPARTEMENT,
    TRANSACTION_YEAR,
    TRANSACTION_MONTH,
    AVERAGE_PRICE_PER_SQUARE_METER,
    CITY_UNIQUE_ID,
)


def compute_city_features(
    transactions_per_city: pl.DataFrame,
    feature_name: str = "ville_demandee",
    grouping_columns: list = [DEPARTEMENT, TRANSACTION_YEAR, TRANSACTION_MONTH],
    quantile_threshold=0.8,
    verbose: bool = False,
):
    nb_transactions_departement = transactions_per_city.group_by(grouping_columns).agg(
        pl.sum(NB_TRANSACTIONS_PER_MONTH).alias("nb_transactions_departement")
    )

    transactions_per_city = transactions_per_city.join(
        nb_transactions_departement,
        on=grouping_columns,
        how="left",
    )

    transactions_per_city = transactions_per_city.with_columns(
        (
            100
            * pl.col(NB_TRANSACTIONS_PER_MONTH)
            / pl.col("nb_transactions_departement")
        ).alias("ratio_transactions_ville")
    ).drop("nb_transactions_departement")

    if verbose:
        print(transactions_per_city.select("ratio_transactions_ville").describe())
    else:
        pass

    transactions_per_city = transactions_per_city.with_columns(
        pl.when(
            pl.col("ratio_transactions_ville")
            > pl.quantile("ratio_transactions_ville", quantile_threshold)
        )
        .then(1)
        .otherwise(0)
        .alias(feature_name),
    ).drop("ratio_transactions_ville")

    return transactions_per_city


def create_debt_ratio_features(
    contexte_macro_eco_annuel: pl.DataFrame,
    debt_ratio_col: str = "taux_endettement",
):
    contexte_macro_eco_annuel = contexte_macro_eco_annuel.with_columns(
        pl.col("date").cast(
            pl.Int32
        ),  # Simple conversion utilisée pour une jointure après
        pl.col(debt_ratio_col).diff().alias("variation_" + debt_ratio_col),
        pl.col(debt_ratio_col).diff().diff().alias("acceleration_" + debt_ratio_col),
    )

    return contexte_macro_eco_annuel


def calculate_interest_rate_features(
    contexte_macro_eco_mensuel: pl.DataFrame,
    aggregation_period: str,
    interest_rate_col: str,
):
    contexte_macro_eco_mensuel = contexte_macro_eco_mensuel.with_columns(
        pl.col(interest_rate_col).diff().name.prefix("variation_"),
        pl.col(interest_rate_col).diff().diff().name.prefix("acceleration_"),
    ).with_columns(
        pl.mean("variation_" + interest_rate_col)
        .rolling(index_column="date", period=aggregation_period)
        .name.prefix("moyenne_glissante_" + aggregation_period + "_"),
    )

    return contexte_macro_eco_mensuel


def compute_price_per_m2_features(
    average_per_month_per_city: pl.DataFrame,
    sort_columns: list = CITY_UNIQUE_ID,
    aggregation_period="6mo",
):
    average_per_month_per_city = (
        average_per_month_per_city.sort(sort_columns)
        .with_columns(
            pl.col(AVERAGE_PRICE_PER_SQUARE_METER)
            .shift()
            .over(CITY_UNIQUE_ID)
            .alias("prix_m2_moyen_mois_precedent"),
            pl.col(NB_TRANSACTIONS_PER_MONTH)
            .shift()
            .over(CITY_UNIQUE_ID)
            .alias("nb_transactions_mois_precedent"),
            pl.col(AVERAGE_PRICE_PER_SQUARE_METER)
            .rolling_mean(window_size=6)
            .over(CITY_UNIQUE_ID)
            .alias("prix_m2_moyen_glissant_" + aggregation_period),
            pl.col(NB_TRANSACTIONS_PER_MONTH)
            .rolling_mean(window_size=6)
            .over(CITY_UNIQUE_ID)
            .alias("nb_transaction_moyen_glissant_" + aggregation_period),
        )
        .filter(
            pl.all_horizontal(
                pl.col(pl.Float32, pl.Float64, pl.Int32, pl.Int64).is_not_nan()
            )
        )
    )

    return average_per_month_per_city


"""'
VERSION DE LA FONCTION UTILISEE DANS LE CADRE DU SCREENCAST P1_C3_FEATURE_ENGINEERING

def compute_price_per_m2_features(
    average_per_month_per_city: pl.DataFrame,
    sort_columns: list = [
        "departement",
        "ville",
        "id_ville",
    ],
):
    average_per_month_per_city = (
        average_per_month_per_city.sort(sort_columns)
        .with_columns(
            pl.col("prix_m2_moyen")
            .shift()
            .over(["departement", "ville", "id_ville"])
            .alias("prix_m2_moyen_mois_precedent"),
            pl.col(NB_TRANSACTIONS_PER_MONTH)
            .shift()
            .over(["departement", "ville", "id_ville"])
            .alias("nb_transactions_mois_precedent"),
        )
        .filter(
            pl.all_horizontal(
                pl.col(pl.Float32, pl.Float64, pl.Int32, pl.Int64).is_not_nan()
            )
        )
    )

    return average_per_month_per_city
"""
