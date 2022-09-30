#Imports libraries necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
import seaborn as sns
scaler = StandardScaler()
enc = LabelEncoder()

from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.svm import SVC,SVR
from sklearn import datasets
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,r2_score,mean_absolute_error,mean_squared_error
from sklearn.cluster import KMeans,AgglomerativeClustering,estimate_bandwidth,MeanShift
import scipy.cluster.hierarchy as sch
# Ignore warnings
#import warnings
#warnings.filterwarnings('ignore')



# %%
# Import the .py files for later calls
import Analyzer as AN
import Classifier as CLA
import Clustering as CLU
import Regressor as RE
#%%
#Loads in the dataset needed
df = pd.read_csv('diamonds.csv')


# Here Begins the Analyzer Portion




# removes NAN values from the dataset
AN.NAN(df)

#Makes a copy of the dataframe to enable categorical graphs later
dfcat=df.copy()

#Drops the column that was made when the dataset was originally created(Remove this line if your dataset does not have this issue)
df = AN.DropCol(df,'Unnamed: 0')

#Encodes the columns provided, simply copy the line and add another column if you need or remove lines as necessary up to the amount of categorical columns
AN.Encode(df,'cut')
AN.Encode(df,'color')
AN.Encode(df,'clarity')

# Here I prepare data for the box plots, this is necessary due to the fact that even after scaling, there seems to be some extreme outliers that make box plots unusable otherwise
all_columns= df.columns

#Grabs the feature columns in preparation for scaling
scale_columns = ['carat','depth', 'table','price','x','y','z']

#Scale the columns of the dataframe based on the inputs above
dfsc = AN.Sca(df,scale_columns)

#Shuffles the data
df1 = AN.Shuf(dfsc)

#Grabs a set amount of the data based on the percentage given (currently set to 70%)
df2= AN.Samp(df1,0.7)

#Gives a resulting CSV file for the prepared data and saves it for future use
df2.to_csv('data_refined.csv',index=False)

#Creates a correlation Matrix of the dataset provided
AN.Cor(df2)

# Creates a Histogram of the previous dataset, you need to input the individual column names of the categorical columns in order to get the graphs requested
AN.HistCat(dfcat,'cut')
AN.HistCat(dfcat,'color')
AN.HistCat(dfcat,'clarity')

#You will need to set some values in the Analyzer.py for this to work properly, check Analyzer.py Line 52 for more information
AN.BoxP(df2,all_columns)

#Creates a pairplot of the sampled dataset
AN.Pair(df2)


# Here Begins the Classification Portion





#Loads XC (X Classifer) as the dataframe without the label and yC(y Classifier) as the label column
XC = df2.drop('cut',axis=1)
yC= df2['cut']

#Performs a train_test_split with a 30% test size
XC_train, XC_test, yC_train, yC_test = train_test_split(XC, yC, test_size=0.30, random_state=0)

# STOP HERE
# %%

# Find the result of the graph and use that to determine your own best values,the numbers refer to the size of the graph, change to your preference/needs
CLA.CKScores(XC_train, XC_test, yC_train, yC_test,1,30)
#%%

#Here you should input the K value you found would work best (in your opinion) as the last value
CLA.KClass(XC_train, XC_test, yC_train, yC_test,8)
#%%

#Gives you the hyperparamters for the next step of the line
CLA.DT_Params(XC_train,yC_train)
#%%

#Creates a Decision Tree Model using the dataset provided, the only parameter you need to possible change is the criterion(currently entropy)
CLA.DecTree(XC_train, XC_test, yC_train, yC_test,'entropy')

#%%

#Gives you hyperparamters for the next step of the line
#If this takes too long please open Classifier.py and read the notes provided there
CLA.RF_Params(XC_train,yC_train)
#%%

#Creates a Random Forest Model using the dataset provided, the only parameter you need to possible change is the criterion(currently gini)
CLA.RanFor(XC_train, XC_test, yC_train, yC_test,'gini')
#%%

#Gives you hyperparamaters for the next step of the line
#If this takes too long please open Classifier.py and read the notes provided there
CLA.SV_Params(XC_train,yC_train)
#%%

#Creates a Support Vector Model using the dataset provided, the only parameters you need to possibly change is the C (1000) and the gamma(0.01)
CLA.SupVec(XC_train, XC_test, yC_train, yC_test,1000,0.01)
#%%

#A word of note, ANN's are particular about the dataset as such I would reccomend fiddling around with the parameters in
#Classifier.py if you dont get a great result, the parameters to change are noted in the Classifier.py itself
CLA.CANN(XC_train, XC_test, yC_train, yC_test)
#%%

#Creates a Linear Classification model using the dataset provided
#Currently disabled due to incompatibility with dataset
#CLA.Cline(XC_train, XC_test, yC_train, yC_test)


#Here Begins the Regression Portion





#%%
#Here starts the Regression Models, but first I prep a new train_test_split 
XR = df2.drop('price',axis=1)
XR

yR= df2['price']
yR


XR_train, XR_test, yR_train, yR_test = train_test_split(XR, yR, test_size=0.30, random_state=0)
#%%

# Find the result of the graph and use that to determine your own best values,the numbers refer to the size of the graph, change to your preference/needs
RE.RKScores(XR_train, XR_test, yR_train, yR_test,1,30)
#%%

#Here you should input the K value you found would work best (in your opinion) as the last value
RE.KReg(XR_train, XR_test, yR_train, yR_test,4)
#%%

#Gives you the hyperparamters for the next step of the line
RE.DT_Params(XR_train,yR_train)
#%%

#Creates a Decision Tree Model using the dataset provided, the only parameters you need to possible change is the criterion(currently squared error)
#and the max_depth(currently 10)
RE.DecTree(XR_train, XR_test, yR_train, yR_test,"squared_error",10)
#%%

#Gives you the hyperparamters for the next step of the line
RE.RF_Params(XR_train,yR_train)
#%%

#Creates a Random Forest Model using the dataset provided, the only parameter you need to possible change is the criterion(currently squared error)
# and the n_estimators(currently 90)
RE.RanFor(XR_train, XR_test, yR_train, yR_test,"squared_error",90)
#%%

#Gives you the hyperparamters for the next step of the line
RE.SV_Params(XR_train,yR_train)
#%%

#Creates a Support Vector Model using the dataset provided, the only parameter you need to possibly change is the C (1000) and the gamma(0.01)
RE.SupVec(XR_train, XR_test, yR_train, yR_test,'auto')
#%%

#A word of note, ANN's are particular about the dataset as such I would recommend fiddling around with the parameters in
#Regressor.py if you dont get a great result, the parameters to change are noted in the Regressor.py itself
RE.RANN(XR_train, XR_test, yR_train, yR_test)
#%%

#Makes a Linear Regression model
#also disabled due to incompatibility
#RE.Rline(XR_train, XR_test, yR_train, yR_test)
#%%
#Here Begins the Clustering Portion




#Prepares a subset of the data without any label columns
XCLU = df2.drop(['cut','color','clarity'],axis=1)

#Makes a graph showing all K values from (currently) 1 to 15 and how well they will work, pick the one you believe is optimal and input it in the next part
CLU.KCluGraph(XCLU,1,15)
#%%

#Makes a cluster using the K value provided, a word of note that you will have to open Clustering.py to change some things dependant on your K Value
#Check Clustering.py for more information
CLU.KClus(XCLU,7)
#%%

#Will make a dendrogram with your X(XCLU) and show everything above a certain limit(140) blue, the color part is purely for show, however feel free to change it
#After you decide on the amount of clusters you want if you'd like it to look nicer
CLU.Den(XCLU,140)
#%%

#Will use Hierarchical Clustering to make clusters for you and print them based on the clusters provided(currently 5), this one has the same requirements as
#KClus above so please check Clustering.py for more information
CLU.HClus(XCLU,5)
#%%

# Uses Mean Shift Clustering to make clusters, you will probably need to run this once first then use the information provided to modify it in Clustering.py
#There is also more information available in Clustering.py
CLU.MSXClus(XCLU,0.3)





















# %%
