def perform_cross_validation(
    X: pl.DataFrame,
    y: pl.Series,
    model,
    cross_val_type,
    scoring_metrics: tuple,
    return_estimator=False,
    groups=None,
):
    scores = cross_validate(
        model,
        X.to_numpy(),
        y.to_numpy(),
        cv=cross_val_type,
        return_train_score=True,
        return_estimator=return_estimator,
        scoring=scoring_metrics,
        groups=groups,
    )

    for metric in scoring_metrics:
        print(
            "Average Train {metric} : {metric_value}".format(
                metric=metric,
                metric_value=np.mean(scores["train_" + metric]),
            )
        )
        print(
            "Train {metric} Standard Deviation : {metric_value}".format(
                metric=metric, metric_value=np.std(scores["train_" + metric])
            )
        )

        print(
            "Average Test {metric} : {metric_value}".format(
                metric=metric, metric_value=np.mean(scores["test_" + metric])
            )
        )
        print(
            "Test {metric} Standard Deviation : {metric_value}".format(
                metric=metric, metric_value=np.std(scores["test_" + metric])
            )
        )

    return scores


# %%
xgboost_classifier = XGBClassifier(random_state=random_state)
classification_scoring_metrics = ("precision", "recall", "roc_auc")

# %%

scores_xgboost = perform_cross_validation(
    X=X[feature_names],
    y=y_classification,
    model=xgboost_classifier,
    cross_val_type=KFold(n_splits=5),
    scoring_metrics=classification_scoring_metrics,
)

# %%
scores_xgboost = perform_cross_validation(
    X=X[feature_names],
    y=y_classification,
    model=xgboost_classifier,
    cross_val_type=StratifiedKFold(n_splits=5),
    scoring_metrics=classification_scoring_metrics,
)
