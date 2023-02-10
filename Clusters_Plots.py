# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 09:58:06 2022

@author: Rotman A. Criollo Manajarrez
"""


########################################################################################
##
##              PLOT CLUSTERS 
##
## Author: Rotman A. Criollo Manajarrez
########################################################################################

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import MiniBatchKMeans 

### reading from the file
filename = 'RES_Sist_Geot_2023.02.09.xlsx'
path = 'F:/RCM/Divulgacion/Articulos/2020 RIGS artiiculo geotermia/'
dfres = pd.read_excel(path + filename, sheet_name='res')


### Adimensionalization

# Scaled by mean and stdrd. deviation
# X = StandardScaler().fit_transform(dfres)

# Scaled by Min and Max values
X = MinMaxScaler().fit_transform(dfres)

# Scaled by its maximum absolute value
# X = MaxAbsScaler().fit_transform(dfres)

# Scaled by its distance between 1st and 3rd quartiles
# X = RobustScaler().fit_transform(dfres)

labels = ['Osorno', 'Santiago de Chile', 
          'Buenos Aires', 'Córdoba', 
          'Barcelona', 'Montevideo', 
          'Quito', 'Zapopan', 
          'Barranquilla', 'Guayaquil']


### Define number of clusters
# n_clusters = 4 #NOTE: include colours  (line 70) as n_clusters you have

# ###### K-MEANS
# kmeans = MiniBatchKMeans(n_clusters = n_clusters)
# kmeans.fit(X)

# # Prepare centroids
# centroids = kmeans.cluster_centers_
# centroids_x = centroids[:,0]
# centroids_y = centroids[:,1]

# # plot kmeans clustering and centroids
# plt.figure(figsize=(10, 7))
# plt.subplots_adjust(bottom=0.1)
# #plot real position of cities
# plt.scatter(X[:,0],X[:,1], label='True Position')
# #plot cluster centroids
# plt.scatter(centroids_x,centroids_y,marker = "x", 
#             s=150,linewidths = 5, zorder = 10,
#             c=['g','r','c','m']) #'b'

# #include city labels 
# for label, x, y in zip(labels, X[:, 0], X[:, 1]):
#     plt.annotate(
#         label,
#         xy=(x, y), xytext=(-3, 3),
#         textcoords='offset points', ha='right', va='bottom')
    
# plt.title('K-MEANS. Climate and soil characteristics at each city', fontsize=15)
# plt.grid(True, color='lightgrey', linestyle='--', linewidth=0.75)
# plt.show()
# # ## info: https://scikit-learn.org/stable/modules/clustering.html#k-means


# ###### PCA
# pca = PCA(n_clusters) 
# pca.fit(X) 
# pca_data = pd.DataFrame(pca.transform(X)) 
# # print(pca_data.head())

# kmeans_pca=KMeans(n_clusters = n_clusters, init="k-means++", random_state=42)
# kmeans_pca.fit(pca_data)
# def show_clusters(data, labels):
#       palette = sns.color_palette('hls', n_colors=len(set(labels)))      
#       sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue=labels, palette=palette)
      
# plt.title('PCA clusters', fontsize=15)
# plt.grid(True, color='lightgrey', linestyle='--', linewidth=0.75)

# show_clusters(pca_data, kmeans_pca.labels_)
## info: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing


###### DENDOGRAMA
# # Types of hierarchical clustering
# ## ‘single’ uses the minimum of the distances between all observations of the two sets.
linked = linkage(X, 'single')

# ## ‘average’ uses the average of the distances of each observation of the two sets.
# linked = linkage(X, 'average')

# ##  ‘complete’ or ‘maximum’ linkage uses the maximum distances between all observations of the two sets.
# linked = linkage(X, 'complete')

# ## ‘ward’ minimizes the variance of the clusters being merged.
# linked = linkage(X, 'ward')

# ## ‘weighted’ Perform weighted/WPGMA linkage on the condensed distance matrix.
# linked = linkage(X, 'weighted')
 
# Plot the dendogram
plt.figure(figsize=(10, 7))
dendrogram(linked,
            p=10,
            truncate_mode = 'lastp',
           orientation='right',
           get_leaves = False, #True,
           labels=labels,
           distance_sort= False, #'descending',
           show_leaf_counts=True)

plt.title('Dendogram. Climate and soil characteristics at each city', fontsize=15)
plt.show()
### info: https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
### info: https://www.askpython.com/python/examples/cluster-analysis-in-python

 

# ############################## BOREHOLE LENGHT ANALYSIS

# df_BHRL = pd.read_excel(path + filename, sheet_name='BHRL_lenght')

# X_BHRL = StandardScaler().fit_transform(df_BHRL)


# ###### K-MEANS
# plt.figure(figsize=(10, 7))
# plt.subplots_adjust(bottom=0.1)
# plt.scatter(X_BHRL[:,0],X_BHRL[:,1], label='True Position')
 
# for label, x, y in zip(labels, X_BHRL[:, 0], X_BHRL[:, 1]):
#     plt.annotate(
#         label,
#         xy=(x, y), xytext=(-3, 3),
#         textcoords='offset points', ha='right', va='bottom')
# plt.title('K-MEANS. Borehole lenght at each city', fontsize=15)
# plt.grid(True, color='lightgrey', linestyle='--', linewidth=0.75)

# plt.show()

# ###### PCA
# pca_BHRL = PCA(4) 
# pca_BHRL.fit(X_BHRL) 
# pca_data_BHRL = pd.DataFrame(pca_BHRL.transform(X_BHRL)) 
# print(pca_data_BHRL.head())

# kmeans_pca_BHRL = KMeans(n_clusters=4, init="k-means++", random_state=42)
# kmeans_pca_BHRL.fit(pca_data_BHRL)

# def show_clusters(data, labels):
#       palette = sns.color_palette('hls', n_colors=len(set(labels)))      
#       sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue=labels,palette=palette)
#       plt.title('PCA clusters 3. Borehole lenght at each city', fontsize=15)
#       plt.grid(True, color='lightgrey', linestyle='--', linewidth=0.75)
      
# show_clusters(pca_data_BHRL, kmeans_pca_BHRL.labels_)


# ###### DENDOGRAMA
 
# linked = linkage(X_BHRL, 'single')
 
# plt.figure(figsize=(10, 7))
# dendrogram(linked,
#             orientation='right',
#             labels=labels,
#             distance_sort= False, #'descending',
#             show_leaf_counts=True)

# plt.title('Dendogram. Borehole lenght at each city', fontsize=15)
# plt.show()
##########################  FIN BOREHOLE ANALYSIS ###########################
