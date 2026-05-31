# %%
# ----------------------------- Imports ----------------------------- #

from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions


# %%
# ----------------------------- Configuration ----------------------------- #

REGRESSION_TARGET = "prix"
CLASSIFICATION_TARGET = "en_dessous_du_marche"

RANDOM_STATE = 42
MAX_POINTS_FOR_DECISION_BOUNDARY = 1200

CURRENT_FILE = Path(__file__).resolve()

# Si le script est dans /screenshots, on remonte à la racine du projet
if CURRENT_FILE.parent.name == "screenshots":
    ROOT_DIR = CURRENT_FILE.parents[1]
else:
    ROOT_DIR = CURRENT_FILE.parent

DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme()


# %%
# ----------------------------- Helpers data ----------------------------- #

def extract_rar_if_needed() -> None:
    """
    Le zip OpenClassrooms contient un fichier .rar.
    Cette fonction tente d'extraire le .rar si le fichier parquet n'est pas encore présent.
    """
    parquet_files = list(DATA_DIR.rglob("transactions_immobilieres.parquet"))

    if parquet_files:
        return

    rar_files = [
        file
        for file in DATA_DIR.rglob("*.rar")
        if "__MACOSX" not in str(file)
    ]

    if not rar_files:
        raise FileNotFoundError(
            "Aucun fichier .rar trouvé dans data/.\n"
            "Vérifie que tu as bien téléchargé puis dézippé les données OpenClassrooms."
        )

    extractor = shutil.which("unar") or shutil.which("unrar")

    if extractor is None:
        raise RuntimeError(
            "Le fichier .rar existe, mais aucun extracteur n'est installé.\n\n"
            "Installe unar avec :\n"
            "brew install unar\n\n"
            "Puis relance :\n"
            "uv run python screenshots/p1_c2.py"
        )

    rar_file = rar_files[0]

    print(f"Extraction du fichier RAR : {rar_file}")

    if Path(extractor).name == "unar":
        command = [extractor, "-f", "-o", str(DATA_DIR), str(rar_file)]
    else:
        command = [extractor, "x", "-o+", str(rar_file), str(DATA_DIR)]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Erreur pendant l'extraction du fichier .rar.")

    print("Extraction terminée.")


def find_transactions_file() -> Path:
    """
    Recherche le fichier transactions_immobilieres.parquet dans data/.
    """
    extract_rar_if_needed()

    parquet_files = list(DATA_DIR.rglob("transactions_immobilieres.parquet"))

    if not parquet_files:
        all_files = "\n".join(str(file) for file in DATA_DIR.rglob("*") if file.is_file())
        raise FileNotFoundError(
            "Impossible de trouver transactions_immobilieres.parquet.\n\n"
            "Fichiers trouvés dans data/ :\n"
            f"{all_files}"
        )

    return parquet_files[0]


def load_transactions() -> pl.DataFrame:
    """
    Charge le dataset principal.
    """
    transactions_file = find_transactions_file()

    print(f"Fichier chargé : {transactions_file}")

    transactions_df = pl.read_parquet(transactions_file)

    print(f"Shape transactions : {transactions_df.shape}")
    print("Colonnes disponibles :")
    for column in transactions_df.columns:
        print(f" - {column}")

    required_columns = [
        "departement",
        "surface_habitable",
        REGRESSION_TARGET,
        CLASSIFICATION_TARGET,
    ]

    missing_columns = [
        column for column in required_columns
        if column not in transactions_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans le dataset : "
            + ", ".join(missing_columns)
        )

    return transactions_df


def filter_departement(df: pl.DataFrame, departement: int | str) -> pl.DataFrame:
    """
    Filtre le département en gérant les cas 4 / 04 / '4' / '04'.
    """
    departement_str = str(departement)
    departement_str_2_digits = departement_str.zfill(2)

    return df.filter(
        pl.col("departement")
        .cast(pl.Utf8)
        .str.strip_chars()
        .is_in([departement_str, departement_str_2_digits])
    )


def sample_for_plot(df: pl.DataFrame, max_rows: int) -> pl.DataFrame:
    """
    Évite que les graphiques de frontière de décision soient trop lourds.
    """
    if df.height <= max_rows:
        return df

    return df.sample(
        n=max_rows,
        seed=RANDOM_STATE,
        shuffle=True,
    )


# %%
# ----------------------------- Chargement des données ----------------------------- #

transactions = load_transactions()

price_quantile_10 = transactions.select(
    pl.col(REGRESSION_TARGET).quantile(0.1)
).item()

print(f"Quantile 10% de {REGRESSION_TARGET} : {price_quantile_10}")


# %%
# ----------------------------- Screenshot Régression Linéaire 2D ----------------------------- #

transactions_regression_2D = (
    filter_departement(transactions, 75)
    .filter(
        pl.col(REGRESSION_TARGET) >= price_quantile_10
    )
    .select(["surface_habitable", REGRESSION_TARGET])
    .drop_nulls()
)

print("\nDataset régression linéaire 2D")
print(transactions_regression_2D.head())
print(f"Shape : {transactions_regression_2D.shape}")

if transactions_regression_2D.height == 0:
    raise ValueError("Aucune donnée disponible pour la régression linéaire.")

X_regression = transactions_regression_2D["surface_habitable"].to_numpy().reshape(-1, 1)
y_regression = transactions_regression_2D[REGRESSION_TARGET].to_numpy()

linear_regressor_2D = LinearRegression()
linear_regressor_2D.fit(X_regression, y_regression)

surface_habitable_range = np.linspace(
    transactions_regression_2D["surface_habitable"].min(),
    transactions_regression_2D["surface_habitable"].max(),
    100,
).reshape(-1, 1)

predictions = linear_regressor_2D.predict(surface_habitable_range)

plt.figure(figsize=(10, 6))
plt.scatter(
    transactions_regression_2D["surface_habitable"],
    transactions_regression_2D[REGRESSION_TARGET],
    label="Bâtiments",
    alpha=0.5,
)

plt.plot(
    surface_habitable_range,
    predictions,
    color="red",
    label="Ligne de régression",
)

plt.xlabel("Surface habitable")
plt.ylabel(REGRESSION_TARGET)
plt.title("Lien entre surface habitable et prix de transaction")
plt.legend()
plt.tight_layout()

linear_output = OUTPUT_DIR / "regression_lineaire_2D.png"
plt.savefig(linear_output, dpi=150)
print(f"Graphique sauvegardé : {linear_output}")

plt.show()


# %%
# ----------------------------- Screenshot Régression Logistique ----------------------------- #

transactions_classification_3D = (
    filter_departement(transactions, 4)
    .filter(
        pl.col(REGRESSION_TARGET) >= price_quantile_10
    )
    .select(["surface_habitable", REGRESSION_TARGET, CLASSIFICATION_TARGET])
    .drop_nulls()
)

transactions_classification_3D = sample_for_plot(
    transactions_classification_3D,
    MAX_POINTS_FOR_DECISION_BOUNDARY,
)

print("\nDataset classification")
print(transactions_classification_3D.head())
print(f"Shape : {transactions_classification_3D.shape}")

if transactions_classification_3D.height == 0:
    raise ValueError("Aucune donnée disponible pour la classification.")

X_classification = transactions_classification_3D.select(
    ["surface_habitable", REGRESSION_TARGET]
).to_numpy()

y_classification_series = transactions_classification_3D[CLASSIFICATION_TARGET]

if y_classification_series.dtype == pl.Boolean:
    y_classification = y_classification_series.cast(pl.Int8).to_numpy()
else:
    y_classification = y_classification_series.to_numpy()

unique_classes = np.unique(y_classification)

if len(unique_classes) < 2:
    raise ValueError(
        "La classification nécessite au moins deux classes différentes.\n"
        f"Classes trouvées : {unique_classes}"
    )

logistic_regressor = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    ),
)

logistic_regressor.fit(X_classification, y_classification)

plt.figure(figsize=(10, 6))
plot_decision_regions(
    X_classification,
    y_classification,
    clf=logistic_regressor,
    legend=2,
)

plt.xlabel("Surface habitable")
plt.ylabel(REGRESSION_TARGET)
plt.title("Decision Boundary de la régression logistique")
plt.tight_layout()

logistic_output = OUTPUT_DIR / "decision_boundary_regression_logistique.png"
plt.savefig(logistic_output, dpi=150)
print(f"Graphique sauvegardé : {logistic_output}")

plt.show()


# %%
# ----------------------------- Screenshot Random Forest ----------------------------- #

rf_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=RANDOM_STATE,
)

rf_classifier.fit(X_classification, y_classification)

plt.figure(figsize=(10, 6))
plot_decision_regions(
    X_classification,
    y_classification,
    clf=rf_classifier,
    legend=2,
)

plt.xlabel("Surface habitable")
plt.ylabel(REGRESSION_TARGET)
plt.title("Decision Boundary de la Random Forest")
plt.tight_layout()

rf_output = OUTPUT_DIR / "decision_boundary_random_forest.png"
plt.savefig(rf_output, dpi=150)
print(f"Graphique sauvegardé : {rf_output}")

plt.show()


# %%
# ----------------------------- Fin ----------------------------- #

print("\nScript terminé avec succès.")
print(f"Graphiques disponibles dans : {OUTPUT_DIR}")