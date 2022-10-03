
# Prepares the libraries for later use.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
import seaborn as sns
scaler = StandardScaler()
enc = LabelEncoder()
# Ignore warnings
#import warnings
#warnings.filterwarnings('ignore')

# Function to remove NAN values
def NAN(data):
    data=data.replace(r'^\s*$', np.nan, regex=True)
    data.dropna(inplace=True,axis=0)
    return

# Function to drop a singular column
def DropCol(data,col):
    data=data.drop([col],axis=1)
    return data

# Function to encode a singular column
def Encode(data,col):
    data[col]=enc.fit_transform(data[col])
    return

# Function to shuffle the dataset
def Shuf(data):
    data = data.sample(frac=1).reset_index(drop=True)
    return data

# Function to provide a correlation matrix
def Cor(data):
    sns.heatmap(data.corr(),cmap='BrBG',annot=True);
    plt.title("Correlation of the Dataset", fontsize =20)
    plt.savefig('Correlation.png', dpi=300, bbox_inches='tight')
    return

# function to create a pairplot
def Pair(data):
    sns.pairplot(data)
    return

# %Function to create a singular histogram
def HistCat(data,col):
    sns.displot(data[col])
    return

# Function to create a mass of subplots, in this case I had 10 columns so I made a 5x2 subplot of them all, if you have more(or less) columns you will need to change this
# from 2, 5 to what you need and also potentially copy+paste/remove more ax's here
def BoxP(data,col):
    fig, axs = plt.subplots(2, 4)
    columns=col

    # basic plot
    axs[0, 0].boxplot(data[col[0]])
    axs[0, 0].set_title(columns[0])

    
    axs[0, 1].boxplot(data[col[1]])
    axs[0, 1].set_title(columns[1])

    
    axs[0, 2].boxplot(data[col[2]])
    axs[0, 2].set_title(columns[2])

    
    axs[0, 3].boxplot(data[col[3]])
    axs[0, 3].set_title(columns[3])

    
    #axs[0, 4].boxplot(data[col[4]])
    #axs[0, 4].set_title(columns[4])

    
    
    axs[1, 0].boxplot(data[col[4]])
    axs[1, 0].set_title(columns[4])

    axs[1, 1].boxplot(data[col[5]])
    axs[1, 1].set_title(columns[5])
    
    axs[1, 2].boxplot(data[col[6]])
    axs[1, 2].set_title(columns[6])
    
    #axs[1, 3].boxplot(data[col[8]])
    #axs[1, 3].set_title(columns[8])
    
    #axs[1, 4].boxplot(data[col[9]])
    #axs[1, 4].set_title(columns[9])
    
    
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,hspace=0.4, wspace=0.3)
                                            
    plt.show()                   
    return

# %%
def Sca(data,col):
    scal_feat = data.copy()
    col_names = col
    feat = scal_feat[col_names]
    scaler = StandardScaler().fit(feat.values)
    feat = scaler.transform(feat.values)
    scal_feat[col_names] = feat
    data = scal_feat
    return data

# %%
def Samp(data,perc):
    data= data.sample(frac = perc)
    return data





