from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks, RepeatedEditedNearestNeighbours

from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc, accuracy_score, precision_score, recall_score

IMBS = {
    'smote': SMOTE(k_neighbors=5),
    'rus': RandomUnderSampler('not minority'),
    'wilson':EditedNearestNeighbours(n_neighbors=5,kind_sel='all'),  #Default was 3
    'tomek': TomekLinks(),
    'None': 'passthrough',
}
CLFS = {
    'dt': DecisionTreeClassifier(max_depth=20),
    'lr': LogisticRegression(solver='lbfgs',max_iter=1000),
    'nb': GaussianNB(),
    'svm': SVC(probability=True),
    'knn': KNeighborsClassifier(n_neighbors=5),
    'rf': RandomForestClassifier(n_estimators=50),
}

bal_nb = GaussianNB()
bal_dt_bag = DecisionTreeClassifier(max_depth=20,max_features='sqrt')
bal_dt_boost = DecisionTreeClassifier(max_depth=10)
ENSEMBLES = {
    'rboost_DT': RUSBoostClassifier(base_estimator=bal_dt_boost,algorithm='SAMME',n_estimators=50),
    'rboost_NB': RUSBoostClassifier(base_estimator=bal_nb,algorithm='SAMME',n_estimators=50),
    'bbag_DT': BalancedBaggingClassifier(base_estimator=bal_dt_bag,n_estimators=50),
    'bbag_NB': BalancedBaggingClassifier(base_estimator=bal_nb,n_estimators=50),
}

CV = RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=99)

def pr_rec_score(y,yp):
    prec, rec, _ = precision_recall_curve(y,yp)
    return auc(rec,prec)
SCORERS = [matthews_corrcoef,pr_rec_score]

