import pandas as pd  
import numpy as np  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.grid_search import GridSearchCV  
from sklearn import cross_validation, metrics  
import matplotlib.pylab as plt  
from sklearn.utils import resample
from sklearn.utils import shuffle 

def train(address):
    train= pd.read_csv(address)  
    target='is_attributed'
    IDcol= 'ip'  
    del train['click_time']
    del train['attributed_time']
    print(train['is_attributed'].value_counts())
    print(type(train))
    df = train
    df_majority = df[df['is_attributed']==0]
    df_minority = df[df['is_attributed']==1]
    major_val = len(df_majority)
    minor_val = len(df_minority)
    
    df_majority_upsampled = resample(df_majority, 
                                     replace=True,    
                                     n_samples=minor_val,   
                                     random_state=123)
    print(len(df_majority_upsampled), " ", minor_val)
    df_upsampled = pd.concat([df_minority, df_majority_upsampled])
    newdata = shuffle(df_upsampled)
    x_columns = [x for x in newdata.columns if x not in [target,IDcol]]  
    X = newdata[x_columns]  
    y = newdata['is_attributed'] 
    rf0 = RandomForestClassifier(oob_score=True, random_state=10)  
    rf0.fit(X,y)  
    print(rf0.oob_score_)
    
    y_predprob = rf0.predict_proba(X)[:,1]  
    print("AUC Score (Train):", metrics.roc_auc_score(y,y_predprob))
    param_test1= {'n_estimators':list(range(10,71,5))}
    print(param_test1)
    gsearch1= GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,  
                                     min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10),  
                           param_grid =param_test1, scoring='roc_auc',cv=5)  
    gsearch1.fit(X,y)  
    print(gsearch1.grid_scores_,gsearch1.best_params_, gsearch1.best_score_)
    param_test2= {'max_depth':list(range(3,10,1)), 'min_samples_split':list(range(50,201,20))}  
    gsearch2= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,  
                                     min_samples_leaf=20,max_features='sqrt' ,oob_score=True,random_state=10),  
       param_grid = param_test2,scoring='roc_auc',iid=False, cv=5)  
    gsearch2.fit(X,y)  
    print(gsearch2.grid_scores_,gsearch2.best_params_, gsearch2.best_score_)