import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer


def research_model(model, params, features, target, cv=5):
    metric = make_scorer(fbeta_score, beta=1, average='macro')
    gscv = GridSearchCV(
        model, params, cv=cv,
        scoring=metric,
        refit=False,
        return_train_score=True,
    )
    gscv.fit(features, target)
    return pd.DataFrame(gscv.cv_results_).sort_values(
        by='mean_test_score', ascending=False
    )
