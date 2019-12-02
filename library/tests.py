import numpy as np
from sklearn.neighbors.base import _get_weights
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, RepeatedStratifiedKFold
from sklearn.datasets import load_iris
from sklearn.ensemble import VotingClassifier

CLFS = [DecisionTreeClassifier(max_leaf_nodes=500), GaussianNB(), KNeighborsClassifier()]
NAMES = [c.__class__.__name__ for c in CLFS]
clf = VotingClassifier(list(zip(NAMES, CLFS)), voting='soft')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2)
X, y = load_iris(return_X_y=True)
b = cross_val_predict(CLFS[0], X, y, cv=cv, method='predict_proba')

