import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors.base import _get_weights
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

def kDN(X, Y, K=5, n_jobs=-1,weight='uniform', **kwargs):
    knn = KNeighborsClassifier(n_neighbors=K, n_jobs=n_jobs, weights=weight).fit(X, Y)
    dist, kid = knn.kneighbors()  # (N,K) : ids & dist of nn's for every sample in X
    weights = _get_weights(dist, weight)
    if weights is None:
        weights = np.ones_like(kid)
    disagreement = Y[kid] != Y.reshape(-1, 1)
    return np.average(disagreement, axis=1, weights=weights)

def Ulta(X, Y, K=5, n_jobs=-1,weight='uniform', **kwargs):
    knn = KNeighborsClassifier(n_neighbors=K, n_jobs=n_jobs, weights=weight).fit(X, Y)
    dist, kid = knn.kneighbors()  # (N,K) : ids & dist of nn's for every sample in X
    print(dist.shape)
    max_dist = np.max(dist,axis=1)
    print(max_dist.shape)
    print(dist[:10],max_dist[:10])

CLFS = [DecisionTreeClassifier(max_leaf_nodes=500), GaussianNB(), KNeighborsClassifier(),
        LogisticRegression(multi_class='auto', max_iter=4000, solver='lbfgs')]

def ih_prob(X, y, n_jobs=1,n_repeats=10, random_state=None):
    NAMES = [c.__class__.__name__ for c in CLFS]
    ests = zip(NAMES,CLFS)
    clf = VotingClassifier(list(ests),voting='soft',n_jobs=n_jobs)
    right_proba = np.zeros_like(y, dtype='float64')
    for it in range(n_repeats):
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=it+1)
        b = cross_val_predict(clf,X,y,cv=cv,method='predict_proba')
        right_proba += b[range(len(X)), y]
    return 1 - right_proba/n_repeats

if __name__=='__main__':
    from sklearn.datasets import load_iris, load_digits, load_breast_cancer
    from sklearn.ensemble import VotingClassifier
    X,y = load_breast_cancer(return_X_y=True)
    a = ih_prob(X,y)
    print(a.mean(),a.std())
    b = gooh(X,y)
    print(b.shape,b.mean(),b.std())