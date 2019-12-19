from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, power_transform
from sklearn.utils import shuffle


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
    df = shuffle(df)
    X = df[all_metrics].values.astype('float64')
    y_noisy = df.HeuBug.values.astype('int8')
    y_real = df.RealBug.values.astype('int8')
    X = np.log1p(X)                      #INFORMATION LEAK, could also use power_transform
    #X = MinMaxScaler().fit_transform(X) #Use of this two transformer needs to be looked at
    assert y_noisy.sum()<len(y_noisy)*.5   #Ensure 1 is bug
    if stats:
        noise = (df.HeuBug!=df.RealBug).sum()/len(df)
        imb = np.unique(y_noisy,return_counts=True)[1]
        print(f"{file} noise:{noise:.3f}, imb:{imb.max()/imb.min():.3f},{imb.min()},{imb.max()}, Shape:{X.shape}")
    return X,y_noisy,y_real

def evaluate(clf,X,y_noisy,y_real,cv,scorers):
    scores = defaultdict(list)
    for train_id, test_id in cv.split(X,y_real):  #vs y_noisy, to solve no-pos-label-in-test-set bug
        clf = clf.fit(X[train_id],y_noisy[train_id])
        probs = clf.predict_proba(X[test_id])
        labels = np.argmax(probs,axis=1)
        for func in scorers:
            yp = probs[:,1]
            try:
                func([0,1,1],[.2,.6,.7])
                yp = probs[:,1]
            except ValueError as e:
                yp = labels
            scores[func.__name__].append(func(y_real[test_id],yp))
    for func in scorers:
        scores[func.__name__] = np.array(scores[func.__name__])
    return scores

# if __name__=='__main__':
#     df = pd.read_csv("../JIRA/groovy-1_5_7.csv")
#     df.drop(columns=["File", 'HeuBugCount', 'RealBugCount'], inplace=True)
#     X = df[all_metrics].values.astype('float64')
#     y_noisy = df.HeuBug.values.astype('int8')
#     y_real = df.RealBug.values.astype('int8')
#     X = np.log1p(X)  # INFORMATION LEAK, could also use power_transform

