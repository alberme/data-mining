from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt
import numpy as np
import pdb
from typing import Tuple

np.set_printoptions(suppress=True)

def create_dist_matrix(coords: np.ndarray) -> np.ndarray:
   row_length = coords.shape[0] #🥱🥱🥱🥱🥱🥱🥱🥱🥱🥱🥱🥱🥱🥱🥱🥱🥱🥱🥱🥱🥱🥱🥱🥱🥱🥱
   dist_triu_indices = np.triu_indices(row_length, 1)
   dist_matrix = np.zeros((row_length, row_length), dtype=np.float32)
   distances = []

   for i in range(row_length - 1):
      x1, y1 = coords[i]
      for x2, y2 in coords[i+1:row_length]:
         distances.append( np.sqrt(np.square(x2 - x1) + np.square(y2 - y1)) )
   dist_matrix[dist_triu_indices] = np.array(distances)

   return dist_matrix + dist_matrix.T


def find_min_node(prox_matrix: np.ndarray) -> Tuple[int, int, float]:
   upper_triangle = np.triu(prox_matrix)    # grab the upper triangle from distance matrix
   # we can replace this line below with a lambda to change distancing methods
   # or do a nasty if elif elif elif statement 🥱
   min_values = upper_triangle[ ~(upper_triangle == 0.0) ].min()  # ignore values equal to 0.0, find minimum values
   min_points = np.vstack(np.where(upper_triangle == min_values))[ :,0 ] # get the indices of those minimum values, stack the two resulting arrays
                                                         # and grab the first column, which will be the first found min_points
   return (*min_points, prox_matrix[ (*min_points,) ])   # combine the min points with min distance, so we get [min_x, min_y, min_distance] 

def update_dist_matrix(prox_matrix: np.ndarray, min_points: tuple) -> np.ndarray:
   result = np.amin(prox_matrix[min_points,:], axis=0) # merges the two min_points rows into one row by min values
   # print(f"\nAfter merging rows {min_points} together\n\t{result}\n")
   ## so yeah, update the matrix correctl
   prox_matrix[ min_points[0] ] = result
   prox_matrix[ :,min_points[0] ] = result
   prox_matrix = np.delete( prox_matrix, min_points[1], axis=0)
   prox_matrix = np.delete( prox_matrix, min_points[1], axis=1)

   return prox_matrix

def merge(min_points: tuple, cluster_children: list, clusters: list, cluster_length: int) -> Tuple[float, float, int]:
   M, N = min_points
   mergedMN = [0.0, 0.0]
   # we need to build [x, y, dist, leaf_count]   
   # yeah you could write this as a list comprehension, but oh well 
   for i,point in enumerate(min_points):
      if len(clusters[point]) > 1:
         mergedMN[i] = cluster_children.index(clusters[point]) + cluster_length # 
      else:
         mergedMN[i] = clusters[point][0] - 1
   
   clusters[M].extend(clusters[N])
   clusters.remove(clusters[N])
   cluster_children.append(clusters[M].copy())
   num_children = len(clusters[M]) # number of leaves or children for this new cluster

   return ( *mergedMN, num_children )

def pair(merges):
   coordinates = []
   for merge in merges:
      coordinates.append([min(merge), max(merge)])
   return coordinates

### this is just to plot the interactive dendrogram and the cluster plot
def plot(linkage_matrix: np.ndarray, data:np.ndarray, interactive=False):
   if interactive:
      plt.ion()
   fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

   titles_axes = ['Dendrogram (Click a point to set distance cut)', 'Cluster']
   titles_xy = [['Index', 'Distance'], ['X', 'Y']]
   fig.suptitle('Hierarchial Clustering')
   color_map = 'RdBu'
   point_size = 5
   average_max = np.mean(linkage_matrix[-2:][:,2]) # this just computes the average value of the two greatest distances
   # just to initially split the dendrogram into two clusters
   dendro_line = axes[0].axhline(average_max, linewidth=0.3, color='r')
   scatter_labels = fcluster(linkage_matrix, average_max, criterion="distance")

   axes[1].scatter(data[:,0], data[:,1], s=point_size, c=scatter_labels, cmap=color_map, alpha=1.0)
   dendrogram(linkage_matrix,
            ax=axes[0],
            truncate_mode="lastp",
            p=10,
            orientation='top',
            distance_sort='descending')

   for i,axe in enumerate(axes):
      x_name, y_name = titles_xy[i]

      axe.set_title(titles_axes[i])
      axe.set(xlabel=x_name, ylabel=y_name)

   # mouse click event handler, so we apply dendrogram cuts to the cluster
   def onclick(event):
      if event.ydata is None:
         return
      scatter_labels = fcluster(linkage_matrix, event.ydata, criterion="distance")
      
      dendro_line.set_ydata(event.ydata)
      dendro_line.set_color('r')
      axes[1].scatter(data[:,0], data[:,1], s=point_size, c=scatter_labels, cmap=color_map, alpha=1.0)
   
   fig.canvas.mpl_connect('button_press_event', onclick)
   plt.show(block=True)
   plt.ioff()
   

"""
Compute the proximity matrix
Let each data point be a cluster
Repeat
	Merge the two closest clusters
	Update the proximity matrix
Until only a single cluster remains
"""

# ### https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial
# # read that tutorial and the juypter notebooks in this folder to understand this stuff 😂🥱

def hierarchial():
   x = [1, 2, 10, 5, 6, 4, 8, 0]
   y = [3, 5, 4, 1, 2, 7, 3, 6]

   data = np.loadtxt("./data/data1.txt", delimiter=",")
   coords = data[:,:2]
   # coords = np.vstack((x, y)).transpose()
   linkage_data_length = coords.shape[0] - 1
   distance_matrix = create_dist_matrix(coords) # distance matrix
   linkage_matrix = [] # the linkage data that explains how the hierarchial tree looks,
   clusters = [[i+1] for i in range(0,coords.shape[0])] # the individual points that we will let be individual clusters
   children = []

   for i in range(linkage_data_length):
      M, N, min_dist = find_min_node(distance_matrix)
      X, Y, num_children = merge(( M,N ), children, clusters, linkage_data_length + 1)
      linkage_matrix.append(( X, Y, min_dist, num_children ))
      distance_matrix = update_dist_matrix(distance_matrix, ( M,N ))

   linkage_matrix = np.array(linkage_matrix, dtype=np.double)
   # coordinates = pair(children)
   # print(linkage_matrix)
   plot(linkage_matrix, coords, interactive=True)

hierarchial()