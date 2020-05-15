import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# iris = load_iris()
# X = iris.data

# # setting distance_threshold=0 ensures we compute the full tree.
# model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage="single")

# model = model.fit(X)
# plt.title('Hierarchical Clustering Dendrogram')
# # plot the top three levels of the dendrogram
# plot_dendrogram(model, truncate_mode='level', p=2)
# plt.xlabel("Number of points in node (or index of point if no parenthesis).")
# plt.ylabel('Distance')
# plt.show()

from sklearn import cluster

np.set_printoptions(suppress=True)
k = 2
color_map = "winter"
karate_adjacency_matrix = np.loadtxt("./data/KarateMatrix.txt")

edges = karate_adjacency_matrix.sum(axis=0) # edge centralities
edge_centrality_matrix = np.identity(karate_adjacency_matrix.shape[0]) * edges
laplacian_matrix = edge_centrality_matrix - karate_adjacency_matrix

eig_values, eig_vectors = np.linalg.eig(laplacian_matrix)

eig_vectors = eig_vectors[:, np.argsort(eig_values)]
eig_values = eig_values[np.argsort(eig_values)]
eig_values[np.argsort(eig_values)]

XX = eig_vectors[:,1:3]

std = np.std(edges)
mean = np.mean(edges)
res  = (edges - mean) / std

print(eig_values[0:5])
eig_vectors[eig_vectors[:,1] > 0]

kmeanz = cluster.KMeans(n_clusters=2, random_state=1234, init='random', verbose=1)
X = kmeanz.fit_transform(karate_adjacency_matrix)
X

## subtract X - X.mean
## normalize X by ^2 then summing axis 1 

## np.sqrt((X * X).sum(axis=1))