
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import datasets
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix,r2_score,mean_absolute_error,mean_squared_error
from sklearn.metrics import accuracy_score

def Pred(model,b):
    y_pred = model.predict(b)
    return y_pred

def Score(model,a,b,c,d):
    test_score = model.score(b,d)
    train_score= model.score(a,c)
    return test_score,train_score

def R2(y_pred,d):
    score = r2_score(d,y_pred)
    return score

def MSError(y_pred,d):    
    MSE = mean_squared_error(d,y_pred)
    return MSE

def MAError(y_pred,d):    
    MAE = mean_absolute_error(d,y_pred)
    return MAE

def RSError(MAE):
    RSE = MAE ** 2
    return RSE

def RKScores(a,b,c,d,e,f):
    scores = []
    for i in range(e,f):
        knn = KNeighborsRegressor(n_neighbors=i)
        knn.fit(a,c)
        score_i = knn.score(b,d)
        scores.append(score_i)
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize =(16,6))
    plt.plot(range(e,f),scores,color = 'green')
    plt.title('Accuracy vs: K Value')
    plt.xlabel('K_Value')
    plt.ylabel('Accuracy')
    return

def KReg(a,b,c,d,e):
    Kmodel = KNeighborsRegressor(n_neighbors=e)
    Kmodel.fit(a, c) 
    KPred= Pred(Kmodel,b)
    KScore = Score(Kmodel,a,b,c,d)
    KR2 = R2(KPred,d)
    KMSE = MSError(KPred,d)
    KMAE = MAError(KPred,d)
    KRMSE = RSError(KMAE)
    print("In order, the test and train scores are:",KScore)
    print('The R2 Score is:',KR2)
    print('The Mean Squared Error is:',KMSE)
    print('The Mean Absolute Error is:',KMAE)
    print('The Root Mean Squared Error is :',KRMSE)
    return

def DT_Params(a,b):
    DTparam_grid = {'max_depth':range(10,50,5), #This is to prevent it from running infinitely, tweak values to higher if score is less than desired(MIGHT help)
                    'criterion':['squared_error','absolute_error'],
                'random_state': [0],
                } 

    DTgrid = GridSearchCV(DecisionTreeRegressor(), DTparam_grid, verbose=2, refit = True)

    DTgrid.fit(a,b)

    print('The best parameters are %s with a score of %0.2f' 
        % (DTgrid.best_params_, DTgrid.best_score_))
    return

def DecTree(a,b,c,d,e,f):
    DTmodel = DecisionTreeRegressor(criterion=e , max_depth = f ,random_state=0)
    DTmodel.fit(a,c)
    DTPred= Pred(DTmodel,b)
    DTScore = Score(DTmodel,a,b,c,d)
    DTR2 = R2(DTPred,d)
    DTMSE = MSError(DTPred,d)
    DTMAE = MAError(DTPred,d)
    DTRMSE = RSError(DTMAE)
    print("In order, the test and train scores are:",DTScore)
    print('The R2 Score is:',DTR2)
    print('The Mean Squared Error is:',DTMSE)
    print('The Mean Absolute Error is:',DTMAE)
    print('The Root Mean Squared Error is :',DTRMSE)
    return

def RF_Params(a,b):
    RFparam_grid = {'n_estimators':range(50,100,10),
                    'criterion':['squared_error','absolute_error'],
                'random_state': [0],
                } 

    RFgrid = GridSearchCV(RandomForestRegressor(), RFparam_grid, verbose=2, refit = True)

    RFgrid.fit(a,b)

    print('The best parameters are %s with a score of %0.2f' 
        % (RFgrid.best_params_, RFgrid.best_score_))
    return

def RanFor(a,b,c,d,e,f):
    RFmodel = RandomForestRegressor(n_estimators=f,criterion=e,random_state=0)
    RFmodel.fit(a, c)
    RFPred= Pred(RFmodel,b)
    RFScore = Score(RFmodel,a,b,c,d)
    RFR2 = R2(RFPred,d)
    RFMSE = MSError(RFPred,d)
    RFMAE = MAError(RFPred,d)
    RFRMSE = RSError(RFMAE)
    print("In order, the test and train scores are:",RFScore)
    print('The R2 Score is:',RFR2)
    print('The Mean Squared Error is:',RFMSE)
    print('The Mean Absolute Error is:',RFMAE)
    print('The Root Mean Squared Error is :',RFRMSE)
    return

def SV_Params(a,b):
    SVparam_grid = {'gamma': ['scale','auto'],
                'kernel': ['rbf'],
                } 

    SVgrid = GridSearchCV(SVR(), SVparam_grid, verbose=2, refit = True) 

    SVgrid.fit(a,b)

    print('The best parameters are %s with a score of %0.2f' 
        % (SVgrid.best_params_, SVgrid.best_score_))
    return

def SupVec(a,b,c,d,e):
    SVmodel = SVR(gamma = e, kernel = 'rbf')
    SVmodel.fit(a,c)
    SVPred= Pred(SVmodel,b)
    SVScore = Score(SVmodel,a,b,c,d)
    SVR2 = R2(SVPred,d)
    SVMSE = MSError(SVPred,d)
    SVMAE = MAError(SVPred,d)
    SVRMSE = RSError(SVMAE)
    print("In order, the test and train scores are:",SVScore)
    print('The R2 Score is:',SVR2)
    print('The Mean Squared Error is:',SVMSE)
    print('The Mean Absolute Error is:',SVMAE)
    print('The Root Mean Squared Error is :',SVRMSE)
    return


def RANN(a,b,c,d):
    ANmodel = MLPRegressor(hidden_layer_sizes=(25,25,25), activation='relu',
                            solver='adam', batch_size=20, max_iter=400, random_state=0)
    ANmodel.fit(a, c)
    ANPred= Pred(ANmodel,b)
    ANScore = Score(ANmodel,a,b,c,d)
    ANR2 = R2(ANPred,d)
    ANMSE = MSError(ANPred,d)
    ANMAE = MAError(ANPred,d)
    ANRMSE = RSError(ANMAE)
    print("In order, the test and train scores are:",ANScore)
    print('The R2 Score is:',ANR2)
    print('The Mean Squared Error is:',ANMSE)
    print('The Mean Absolute Error is:',ANMAE)
    print('The Root Mean Squared Error is :',ANRMSE)
    return

def Rline(a,b,c,d): 
    Lmodel = LinearRegression()
    Lmodel.fit(a, c)
    LPred= Pred(Lmodel,b)
    LScore = Score(Lmodel,a,b,c,d)
    LR2 = R2(LPred,d)
    LMSE = MSError(LPred,d)
    LMAE = MAError(LPred,d)
    LRMSE = RSError(LMAE)
    print("In order, the test and train scores are:",LScore)
    print('The R2 Score is:',LR2)
    print('The Mean Squared Error is:',LMSE)
    print('The Mean Absolute Error is:',LMAE)
    print('The Root Mean Squared Error is :',LRMSE)
    return
