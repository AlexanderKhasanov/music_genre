from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from numpy.random import RandomState

RANDOM_STATE = RandomState(42)

ENCODER_LIMIT = OneHotEncoder(drop='first', dtype='int')
ENCODER_UNLIMIT = OneHotEncoder(handle_unknown='ignore', dtype='int')

MODEL_LOG_REGRESSION = LogisticRegression(
    max_iter=500, random_state=RANDOM_STATE, n_jobs=-1
)
MODEL_TREE = DecisionTreeClassifier(random_state=RANDOM_STATE)
MODEL_RAND_FOREST = RandomForestClassifier(
    random_state=RANDOM_STATE, n_jobs=-1,
)
MODEL_KNN = KNeighborsClassifier(n_jobs=-1)

GSCV_LOG_REGRESSION_CONFIG = {
    'C': [.1, 1, 10, 25, 50, 100, 250, 500, 750, 1000],
    'solver': ['lbfgs', 'liblinear'],
    'penalty': ['None', 'l1', 'l2', 'elasticnet'],
}
GSCV_TREE_CONFIG = {
    'max_depth': range(1, 11)
}
GSCV_RAND_FOREST_CONFIG = {
    'max_depth': range(1, 11),
    'n_estimators': range(1, 202, 5),
}
GSCV_KNN = {
    'n_neighbors': range(2, 21)
}
