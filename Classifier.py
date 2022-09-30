# Loads the libraries necessary for it to work

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,r2_score

# Predicts based on model input
def Pred(model,test):
    y_pred = model.predict(test)
    return y_pred

# Scores based on model input
def Score(model,a,b,c,d):
    test_score = model.score(b,d)
    train_score= model.score(a,c)
    return test_score,train_score


def CM(predictions,e):
    matrix = confusion_matrix(e, predictions)
    find = sns.heatmap(matrix, annot=True, cmap='Blues',fmt = ".1f")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    return plt.show(find)


def CKScores(a,b,c,d,e,f):
    scores = []
    for i in range(e,f):
        knn = KNeighborsClassifier(n_neighbors=i)
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

def KClass(a,b,c,d,e):
    
    Kmodel = KNeighborsClassifier(n_neighbors=e)
    Kmodel.fit(a, c) 
    KPred= Pred(Kmodel,b)
    KScore = Score(Kmodel,a,b,c,d)
    KCM= CM(KPred,d)
    print("In order, the test and train scores are:",KScore)
    print(KCM)
    return

def DT_Params(a,b):
    DTparam_grid = {'criterion': ['gini','entropy'],
                'random_state': [0],
                'splitter': ['best'],  
                } 

    #Prepares the actual Grid Search CV with its own hyperparamers
    DTgrid = GridSearchCV(DecisionTreeClassifier(), DTparam_grid, verbose=2, refit = True)

    #Runs it with the XC_train and yC_train
    DTgrid.fit(a,b)

    #Prints the Best Parameters for you to use in the next cell
    print('The best parameters are %s with a score of %0.2f' 
        % (DTgrid.best_params_, DTgrid.best_score_))
    return

def DecTree(a,b,c,d,e):
    DTmodel = DecisionTreeClassifier(criterion=e , splitter='best',random_state=0)
    DTmodel.fit(a,c)
    DTPred= Pred(DTmodel,b)
    DTScore = Score(DTmodel,a,b,c,d)
    DTCM= CM(DTPred,d)
    print("In order, the test and train scores are:",DTScore)
    print(DTCM)
    return

#If this runs for too long use max_depth, for more information please look up Random Forest Classifier in the sklearn database
def RF_Params(a,b):
    RFparam_grid = {'criterion': ['gini','entropy'],
                'random_state': [0]
                
                } 

    #Prepares the actual Grid Search CV with its own hyperparamers
    RFgrid = GridSearchCV(RandomForestClassifier(), RFparam_grid, verbose=2, refit = True)

    #Runs it with the XC_train and yC_train
    RFgrid.fit(a,b)

    #Prints the Best Parameters for you to use in the next cell
    print('The best parameters are %s with a score of %0.2f' 
        % (RFgrid.best_params_, RFgrid.best_score_))
    return

def RanFor(a,b,c,d,e):
    RFmodel = RandomForestClassifier(criterion=e,random_state = 0)
    RFmodel.fit(a,c)
    RFPred= Pred(RFmodel,b)
    RFScore = Score(RFmodel,a,b,c,d)
    RFCM= CM(RFPred,d)
    print("In order, the test and train scores are:",RFScore)
    print(RFCM)
    return

def SV_Params(a,b):
    #Preparing the Support Vector Parameters
    # You can add Linear and Poly if you'd like to the kernel, however I have left them out to save time as my computer requires ~5 minutes for each fit
    #Another time saver is to add max_iter to the list as by doing so you can limit how many times it will try to optimize the line of best fit
    SVparam_grid = {'C': [0.1,1, 10, 100, 1000],
                'gamma': [1,0.1,0.01,0.001,0.0001],
                'kernel': ['rbf'],
                'random_state': [0]} 

    # Prepares the actual Grid Search CV with its own hyperparamers
    SVgrid = GridSearchCV(SVC(), SVparam_grid, verbose=2, refit = True) 

    # Runs it with the XC_train and yC_train
    SVgrid.fit(a,b)

    # Prints the Best Parameters for you to use in the next cell
    print('The best parameters are %s with a score of %0.2f' 
        % (SVgrid.best_params_, SVgrid.best_score_))
    return

def SupVec(a,b,c,d,e,f):
    SVmodel = SVC(C = e, gamma = f, kernel = 'rbf')
    SVmodel.fit(a, c)
    SVPred= Pred(SVmodel,b)
    SVScore = Score(SVmodel,a,b,c,d)
    SVCM= CM(SVPred,d)
    print("In order, the test and train scores are:",SVScore)
    print(SVCM)
    return

# A quick note here, I am unable to make this work with keras classifiers due to the fact I label encoded the data, however it would allow for 
# a more complicated and possibly more accurate model as such should you have one that works with label encoded data 
# feel free to input it yourself
def CANN(a,b,c,d):
    # Finds the amount of unique labels found in the yC_train
    classes = len(np.unique(b,return_index=False,return_counts=False))

    #Creates an ANN Classification Model with layers of 50,50 and a layer with the amount of classes as its output, you may have to change the 
    #numbers in both the hidden layers(the 50 ones) and the iterations/batches to obtain a more satisfactory result
    ANmodel = MLPClassifier(hidden_layer_sizes=(50,50,classes), activation='relu',
                        solver='adam', batch_size=40, max_iter=50, random_state=0,verbose=True)

    #Runs the model on your data
    ANmodel.fit(a, c) 
    ANPred= Pred(ANmodel,b)
    ANScore = Score(ANmodel,a,b,c,d)
    ANCM= CM(ANPred,d)
    print("In order, the test and train scores are:",ANScore)
    print(ANCM)
    return


def Cline(a,b,c,d): 
    Lmodel = LogisticRegression()
    Lmodel.fit(a, c)
    LPred= Pred(Lmodel,b)
    LScore = Score(Lmodel,a,b,c,d)
    LCM= CM(LPred,d)
    print("In order, the test and train scores are:",LScore)
    print(LCM)
    return