from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.metrics import geometric_mean_score


file = """AvgCyclomatic, AvgCyclomaticModified, AvgCyclomaticStrict, AvgEssential, AvgLine, AvgLineBlank, AvgLineCode, AvgLineComment, CountDeclClass, CountDeclClassMethod, CountDeclClassVariable, CountDeclFunction, CountDeclInstanceMethod,
CountDeclInstanceVariable, CountDeclMethod, CountDeclMethodDefault, CountDeclMethodPrivate, CountDeclMethodProtected,
CountDeclMethodPublic, CountLine, CountLineBlank, CountLineCode, CountLineCodeDecl, CountLineCodeExe, CountLineComment, CountSemicolon, CountStmt, CountStmtDecl, CountStmtExe, MaxCyclomatic, MaxCyclomaticModified, MaxCyclomaticStrict, RatioCommentToCode, SumCyclomatic, SumCyclomaticModified, SumCyclomaticStrict, SumEssential"""
cls = """CountClassBase, CountClassCoupled, CountClassDerived, MaxInheritanceTree, PercentLackOfCohesion"""
meth_prefix = ["CountInput","CountOutput","CountPath","MaxNesting"]

file_metrics = [c.strip() for c in file.split(',')]
cls_metrics = [c.strip() for c in cls.split(',')]
meth_metrics = ['CountInput_Max', 'CountInput_Mean', 'CountInput_Min','CountOutput_Max','CountOutput_Mean',
 'CountOutput_Min','CountPath_Max','CountPath_Mean','CountPath_Min','MaxNesting_Max','MaxNesting_Mean','MaxNesting_Min']
code_metrics = set(file_metrics) | set(cls_metrics) | set(meth_metrics)
process_metrics = ["COMM","Added_lines","Del_lines","ADEV","DDEV"]
own_metrics = ["OWN_LINE","OWN_COMMIT","MINOR_LINE","MINOR_COMMIT","MAJOR_COMMIT","MAJOR_LINE"]
all_metrics = set(code_metrics) | set(own_metrics) | set(process_metrics)

def read_data(file,stats=True):
    df = pd.read_csv("JIRA/"+file)
    df.drop(columns=["File",'HeuBugCount','RealBugCount'],inplace=True)
    X = df[all_metrics].values.astype('float64')
    y_noisy = df.HeuBug.values.astype('int8')
    y_real = df.RealBug.values.astype('int8')
    X = MinMaxScaler().fit_transform(X)
    assert y_noisy.sum()<len(y_noisy)*.5   #Ensure 1 is bug
    if stats:
        noise = (df.HeuBug!=df.RealBug).sum()/len(df)
        imb = np.unique(y_noisy,return_counts=True)[1]
        print(f"noise:{noise}, imb:{imb.max()/imb.min():.3f},{imb.min()},{imb.max()}, Shape:{X.shape}")
    return X,y_noisy,y_real


def evaluate(clf,X,y_noisy,y_real,cv):
    scores = defaultdict(list)
    for train_id, test_id in cv.split(X,y_noisy):
        clf = clf.fit(X[train_id],y_noisy[train_id])
        pred = clf.predict(X[test_id])
        scores['auc'].append(roc_auc_score(y_real[test_id],pred))
        scores['gmean'].append(geometric_mean_score(y_real[test_id],pred))
        #print(scores['auc'][-1],scores['gmean'][-1])
    scores['auc'] = np.array(scores['auc'])
    scores['gmean'] = np.array(scores['gmean'])
    return scores['auc'].mean(),scores['auc'].std(),scores['gmean'].mean(),scores['gmean'].std()