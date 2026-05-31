# Maitrisez l'apprentissage supervisé


## Contexte

Ce repository centralise toutes les ressources (code de screencasts, énoncés et corrigés d'exercices) du projet filé lié au cours "Maitrisez l'apprentissage supervisé". La structure des différents dossiers est comme suit

* **exercices :**  Notebooks d'énoncés et de corrigés pour chaque chapitre du cours
* **screencasts :**  Code présenté pendant chaque screencast
* **screenshots :**  Code permettant de générer certains screenshots du cours


## Comment utiliser ce repository

### Téléchargement des données

Vous pouvez télécharger tous les jeux de données dont vous aurez besoin via [ce zip](https://s3.eu-west-1.amazonaws.com/course.oc-static.com/courses/8431846/donnees_maitrisez_apprentissage_supervise.rar.zip). Vous y trouverez :
* **transactions_immobilieres.parquet** : le fichier de données principal qui servira pour toute la partie 1 
* **transactions_par_ville.parquet** : fichier de données contenant des données agrégées à la maille ville, très utilisé dans toutes les parties du cours pour le calcul de features ou de la target
* **transactions_post_feature_engineering.parquet** : fichier de données utilisé dans la partie 2 et 3, où le feature engineering est considéré acquis. 
* **transactions_extra_infos.parquet** : fichier de données contenant des informations sur les observations qui ne peuvent pas être utilisées comme features (id_transaction, date_transaction) mais qui sont utiles pour des analyses exploratoires. A utiliser avec le fichier transactions_post_feature_engineering.parquet
* **features_used.json** : fichier sous forme de liste contenant toutes les features utilisées dans la modélisation. 
* **categorical_features_used.json** : sous-ensemble du ficher précédent, avec uniquement les features qualitatives 

Les différentes données sont soit au format parquet (format plus pratique que le csv et utilisable avec Pandas ou Polars) soit au format JSON concernant des listes de features.

Dans le cas où vous souhaiteriez reconstituer la donnée transactions_immobilieres.parquet, vous pouvez partir des données brutes dans [ce zip](https://s3.eu-west-1.amazonaws.com/course.oc-static.com/courses/8431846/cours_supervise_brut.rar.zip ) et utiliser le script preprocessing.py.

Pour ne pas alourdir le repo, les données et les modèles sauvegardés via Mlflow n'ont pas été chargés ici. Pour ces derniers, vous avez à disposition le code pour les reconstruire. 

### Informations importantes sur les données

Chaque ville dans le jeu de donnée peut être identifié de manière unique via une combinaison des colonnes : id_ville, ville et département 

Concernant la colonne target de classification, elle est calculée de la manière suivante : 
* On calcule prix moyen au m2 à l'échelle départementale, pour un mois donné
* Pour chaque transaction, si son prix au m2 est inférieur à la moyenne de plus de 10%, la target aura comme valeur 1 
* Sinon, la target aura comme valeur 0  


### Comment utiliser le code


**Ce repo utilise Poetry pour le package management et la création d'un environnement virtuel.** le fichier pyproject.toml contient toutes les versions de packages utilisées dans le cours. 

Pour des raisons de conflits de dépendances avec Mlflow, le package BentoML (utilisé pour la partie 3 chaptire 2) ne figure pas parmi les dépendences installées. Toutefois, vous trouverez en commentaire dans le fichier pyprojet.toml la version exacte qui a été utilisée pour le code des screencasts et des exercices.

Pour les exercices, nous vous recommandons de stocker toutes les données au même endroit et de définir cette adresse **via une variable d'environnement** (comme dans le fichier setting.py). Vous pouvez implémenter cela en suivant [ce tutoriel](https://www.youtube.com/watch?v=IolxqkL7cD8&t=201s) par exemple

