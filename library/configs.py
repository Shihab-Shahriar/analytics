from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks

from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc

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

CV = RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=99)

def pr_rec_score(y,yp):
    prec, rec, _ = precision_recall_curve(y,yp)
    return auc(rec,prec)
SCORERS = [matthews_corrcoef,pr_rec_score]
