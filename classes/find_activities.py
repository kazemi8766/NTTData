import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import warnings
warnings.filterwarnings('ignore')

class find_activities:

    def __init__(self,
                 data,
                 feature_list: list,
                 max_nc: int,
                 method=None):
        self.__feature_list = feature_list  # list of the features needs for clustring
        self.__max_nc = max_nc  # maximun number of clusters
        self.__data = data  # preProcessing result
        self.__method = method

    def prepare_data(self):
        X = (pd.concat(self.__data)).reset_index(drop=True)[self.__feature_list].values
        return X

    def __elbow_method(self, X):
        wcss = []  # Within-Cluster Sum of Squares
        for i in range(1, self.__max_nc):
            find_centroids = KMeans(n_clusters=i, random_state=0)
            find_centroids.fit(X)
            wcss.append(find_centroids.inertia_)
        plt.figure(figsize=(9, 7))
        plt.plot(range(1, self.__max_nc), wcss)
        plt.title('Elbow Method\n')
        plt.xlabel('\n N Clusters')
        plt.ylabel('Within-Cluster Sum of Squares \n(WCSS)')
        plt.show()

    def __silhouette(self, X):
        for n_clus in range(2, self.__max_nc):
            clu = KMeans(n_clusters=n_clus, random_state=0)
            cluster_lable = clu.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_lable)
        max_value = max(silhouette_avg)
        max_sil = silhouette_avg.index(max_value)
        return max_sil

    def max_k(self, X):
        if self.__method == None:
            self.__elbow_method(X)
        else:
            max_sil = self.__silhouette(X)
            return max_sil

    def kmeans_cluster(self, k_max, X):
        """
        @param k_max: Best number of cluster for kemans
        @param data: The data we are going to use as the imput
        @return: dataframe with lable
        """
        km = KMeans(n_clusters=k_max,  # K is the best silhouette_score or elbow method came from max_k function
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    tol=1e-04,
                    random_state=0).fit(X)
        cluster_km = km.labels_
        return cluster_km