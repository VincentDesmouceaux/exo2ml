# %%

# ----------------------------- Screenshot Regression Lineaire 2D ----------------------------- #

REGRESSION_TARGET = "prix"
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import polars as pl


"""
On se restreint au périmètre du screenshot; 
Ici nous utilisons la syntaxe de Polars. Mais un équivalent Pandas serait

transactions_regression_2D = transacations[
    (transactions["departement"] == 75)
    & (transactions[REGRESSION_TARGET] >= transactions[REGRESSION_TARGET].quantile(0.1))
][["surface_habitable", REGRESSION_TARGET]]
"""

transactions_regression_2D = transactions.filter(
    pl.col("departement") == 75,
    pl.col(REGRESSION_TARGET) >= pl.quantile(REGRESSION_TARGET, 0.1),
).select(["surface_habitable", REGRESSION_TARGET])

# On crée un modèle de régression linéaire qui essaie d'expliquer la target à partir de la feature choisie
linear_regressor_2D = LinearRegression()

"""
A l'heure d'écriture de ce code, Scikit-learn ne supporte pas les Polars DataFrame directement. 
Il faut alors réaliser une conversion en numpy array, le format au coeur de la majorité des opération Scikit-learn aujourd'hui
"""
linear_regressor_2D.fit(
    transactions_regression_2D["surface_habitable"].to_numpy().reshape(-1, 1),
    transactions_regression_2D[REGRESSION_TARGET].to_numpy(),
)

# On trace un nuage de points classique des batiments
plt.scatter(
    transactions_regression_2D["surface_habitable"],
    transactions_regression_2D[REGRESSION_TARGET],
    label="Batiments",
)

# Simulation de points factices, simplement pour générer la droite
surface_habitable_range = np.linspace(
    transactions_regression_2D["surface_habitable"].min(),
    transactions_regression_2D["surface_habitable"].max(),
    100,
).reshape(-1, 1)
predictions = linear_regressor_2D.predict(surface_habitable_range)

# Ajout de la ligne de regression
plt.plot(surface_habitable_range, predictions, color="red", label="Ligne de regression")
plt.xlabel("Surface habitable")
plt.ylabel(REGRESSION_TARGET)
plt.title("Lien entre surface habitable et prix de transaction")
plt.legend()
plt.show()

# %%
# ---------------------- Screenshot Regression Logistique Frontière de Décision ---------------------- #

CLASSIFICATION_TARGET = "en_dessous_du_marche"
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LogisticRegression

# Nous choissons ces 2 features uniquement pour illustrer le concept !
transactions_classification_3D = transactions.filter(
    pl.col("departement") == 4,
    pl.col(REGRESSION_TARGET) >= pl.quantile(REGRESSION_TARGET, 0.1),
).select(["surface_habitable", REGRESSION_TARGET, CLASSIFICATION_TARGET])

# On entraine notre regression logistique
logistic_regressor = LogisticRegression()
X = transactions_classification_3D.select(["surface_habitable", "prix"]).to_numpy()
y = transactions_classification_3D[CLASSIFICATION_TARGET].to_numpy()
logistic_regressor.fit(X, y)


# Nous faisons appel à cette librairie pour tracer la frontière de décision
plot_decision_regions(X, y, clf=logistic_regressor, legend=2)
plt.xlabel("Surface habitable")
plt.ylabel(REGRESSION_TARGET)
plt.title("Decision Boundary de la régression logistique")
plt.legend()
plt.show()


# ----------------- Idem pour le modèle Random Forest ----------------- #

from sklearn.ensemble import RandomForestClassifier

from mlxtend.plotting import plot_decision_regions

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X, y)

plot_decision_regions(X, y, clf=rf_classifier, legend=2)
plt.xlabel("Surface habitable")
plt.ylabel("prix")
plt.title("Decision Boundary de la Random Forest")
plt.legend()
plt.show()
