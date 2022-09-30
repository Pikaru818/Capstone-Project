
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans,AgglomerativeClustering,estimate_bandwidth,MeanShift
import scipy.cluster.hierarchy as sch
import warnings
warnings.filterwarnings('ignore')

def KCluGraph(a,b,c):
    scores = []
    for i in range(b, c):
        kmodel = KMeans(n_clusters=i, random_state=0)
        kmodel.fit(a)
        scores.append(kmodel.inertia_)

    import matplotlib.pyplot as plt
    plt.plot(range(1, 15), scores, marker='.', markersize=10)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    return

#Here you will have to add extra sns.scatterplot instances to match the amount of K that you input(currently 7)
def KClus(a,b):
    kmeans = KMeans(n_clusters=b, tol = 0.01,random_state = 0)
    kmeans.fit(a)
    Ky = kmeans.predict(a)
    print(kmeans.inertia_)
    plt.figure(figsize=(15,7))
    sns.scatterplot(a.iloc[Ky == 0, 0], a.iloc[Ky == 0, 1], color = 'yellow', label = 'Cluster 1',s=30)
    sns.scatterplot(a.iloc[Ky == 1, 0], a.iloc[Ky == 1, 1], color = 'blue', label = 'Cluster 2',s=30)
    sns.scatterplot(a.iloc[Ky == 2, 0], a.iloc[Ky == 2, 1], color = 'green', label = 'Cluster 3',s=30)
    sns.scatterplot(a.iloc[Ky == 3, 0], a.iloc[Ky == 3, 1], color = 'grey', label = 'Cluster 4',s=30)
    sns.scatterplot(a.iloc[Ky == 4, 0], a.iloc[Ky == 4, 1], color = 'orange', label = 'Cluster 5',s=30)
    sns.scatterplot(a.iloc[Ky == 5, 0], a.iloc[Ky == 5, 1], color = 'teal', label = 'Cluster 6',s=30)
    sns.scatterplot(a.iloc[Ky == 6, 0], a.iloc[Ky == 6, 1], color = 'black', label = 'Cluster 7',s=30)
    sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red', 
                    label = 'Centroids',s=100,marker='d')
    plt.grid(False)
    plt.title('Kmeans Clustering')
    plt.legend()
    plt.show()
    return

def Den(a,b):
    plt.figure(figsize=(18,6))
    plt.title('Dendrogram')
    plt.xlabel('Individual Features')
    plt.ylabel('Euclidean distances')
    dendrogram = sch.dendrogram(sch.linkage(a, method ='ward'),
                                color_threshold=b, 
                                above_threshold_color='blue')
    return

#Here you will have to add extra sns.scatterplot instances to match the amount of n_clusters that you input(currently 5)
def HClus(a,b):
    Agg = AgglomerativeClustering(n_clusters=b, linkage='average')
    Agg.fit(a)
    Ay = Agg.fit_predict(a)
    plt.figure(figsize=(15,7))
    sns.scatterplot(a.iloc[Ay == 0, 0], a.iloc[Ay == 0, 1], color = 'yellow', label = 'Cluster 1',s=50)
    sns.scatterplot(a.iloc[Ay == 1, 0], a.iloc[Ay == 1, 1], color = 'blue', label = 'Cluster 2',s=50)
    sns.scatterplot(a.iloc[Ay == 2, 0], a.iloc[Ay == 2, 1], color = 'green', label = 'Cluster 3',s=50)
    sns.scatterplot(a.iloc[Ay == 3, 0], a.iloc[Ay == 3, 1], color = 'grey', label = 'Cluster 4',s=50)
    sns.scatterplot(a.iloc[Ay == 4, 0], a.iloc[Ay == 4, 1], color = 'orange', label = 'Cluster 5',s=50)
    plt.grid(False)
    plt.title('Hierarchical Clustering')
    plt.legend()
    plt.show()
    return


def MSXClus(a,b):
    Xbandwidth = estimate_bandwidth(a, quantile=b)
    # While estimate_bandwidth works wonders, for this case I think that mean shift clustering just doesnt work all that well, 
    # since it ends up having to make so many clusters in order to work correctly that it ends up making way too many individual clusters,
    # although in theory that means it managed to split it into that many  classes, that also means that in a graph, that really doesnt show up all that well


    #Here you will have to add extra sns.scatterplot instances to match the amount of clusters that it provides(the red diamonds on the graph) so it 
    #requires a lot of tweaking to get it working properly, this is probably not optimized in the slightest but as any other quantile gives way 
    #too many or little clusters, I left it as is
    X_meanshift = MeanShift(bandwidth = Xbandwidth)
    X_meanshift.fit(a)
    yMdf = X_meanshift.predict(a)
    X=a
    plt.figure(figsize=(15,7))
    sns.scatterplot(X.iloc[yMdf == 0, 0], X.iloc[yMdf == 0, 1], color = 'yellow', label = 'Cluster 1',s=50)
    sns.scatterplot(X.iloc[yMdf == 1, 0], X.iloc[yMdf == 1, 1], color = 'blue', label = 'Cluster 2',s=50)
    sns.scatterplot(X.iloc[yMdf == 2, 0], X.iloc[yMdf == 2, 1], color = 'green', label = 'Cluster 3',s=50)
    sns.scatterplot(X.iloc[yMdf == 3, 0], X.iloc[yMdf == 3, 1], color = 'grey', label = 'Cluster 4',s=50)
    sns.scatterplot(X.iloc[yMdf == 4, 0], X.iloc[yMdf == 4, 1], color = 'teal', label = 'Cluster 5',s=50)
    sns.scatterplot(X.iloc[yMdf == 5, 0], X.iloc[yMdf == 5, 1], color = 'pink', label = 'Cluster 6',s=50)
    sns.scatterplot(X.iloc[yMdf == 6, 0], X.iloc[yMdf == 6, 1], color = 'purple', label = 'Cluster 7',s=50)
    sns.scatterplot(X.iloc[yMdf == 7, 0], X.iloc[yMdf == 7, 1], color = 'orange', label = 'Cluster 8',s=50)
    sns.scatterplot(X.iloc[yMdf == 8, 0], X.iloc[yMdf == 8, 1], color = 'cyan', label = 'Cluster 9',s=50)
    sns.scatterplot(X.iloc[yMdf == 9, 0], X.iloc[yMdf == 9, 1], color = 'brown', label = 'Cluster 10',s=50)
    sns.scatterplot(X.iloc[yMdf == 10, 0], X.iloc[yMdf == 10, 1], color = 'black', label = 'Cluster 11',s=50)
    sns.scatterplot(X.iloc[yMdf == 11, 0], X.iloc[yMdf == 11, 1], color = 'grey', label = 'Cluster 12',s=50)
    sns.scatterplot(X.iloc[yMdf == 12, 0], X.iloc[yMdf == 12, 1], color = 'tan', label = 'Cluster 13',s=50)
    sns.scatterplot(X.iloc[yMdf == 13, 0], X.iloc[yMdf == 13, 1], color = 'olive', label = 'Cluster 14',s=50)
    sns.scatterplot(X.iloc[yMdf == 14, 0], X.iloc[yMdf == 14, 1], color = 'lime', label = 'Cluster 15',s=50)
    sns.scatterplot(X.iloc[yMdf == 15, 0], X.iloc[yMdf == 15, 1], color = 'fuchsia', label = 'Cluster 16',s=50)
    sns.scatterplot(X.iloc[yMdf == 16, 0], X.iloc[yMdf == 16, 1], color = 'crimson', label = 'Cluster 17',s=50)
    sns.scatterplot(X.iloc[yMdf == 17, 0], X.iloc[yMdf == 17, 1], color = 'goldenrod', label = 'Cluster 18',s=50)
    sns.scatterplot(X.iloc[yMdf == 18, 0], X.iloc[yMdf == 18, 1], color = 'indigo', label = 'Cluster 19',s=50)
    sns.scatterplot(X.iloc[yMdf == 19, 0], X.iloc[yMdf == 19, 1], color = 'salmon', label = 'Cluster 20',s=50)
    sns.scatterplot(X.iloc[yMdf == 20, 0], X.iloc[yMdf == 20, 1], color = 'sandybrown', label = 'Cluster 21',s=50)
    sns.scatterplot(X.iloc[yMdf == 21, 0], X.iloc[yMdf == 21, 1], color = 'cadetblue', label = 'Cluster 22',s=50)
    sns.scatterplot(X.iloc[yMdf == 22, 0], X.iloc[yMdf == 22, 1], color = 'violet', label = 'Cluster 23',s=50)
    sns.scatterplot(X.iloc[yMdf == 23, 0], X.iloc[yMdf == 23, 1], color = 'aquamarine', label = 'Cluster 24',s=50)
    sns.scatterplot(X_meanshift.cluster_centers_[:, 0], X_meanshift.cluster_centers_[:, 1], color = 'red', 
                    label = 'Centroids',s=100,marker='d')
    plt.grid(False)
    plt.title('Mean Shift Clustering')
    plt.legend()
    plt.show()
    return



