from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
# from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

np.set_printoptions(suppress=True)

def create_dist_matrix(coords: np.ndarray) -> np.ndarray:
  row_length = coords.shape[0] 
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
  try:
    min_values = upper_triangle[ ~(upper_triangle == 0.0) ].min()  # ignore values equal to 0.0, find minimum values
  except ValueError:
    min_values = np.zeros(upper_triangle.shape[0])
  min_points = np.vstack(np.where(upper_triangle == min_values))[ :,0 ] # get the indices of those minimum values, stack the two resulting arrays

  return (*min_points, prox_matrix[ (*min_points,) ])   # combine the min points with min distance, so we get [min_x, min_y, min_distance] 

def update_dist_matrix(prox_matrix: np.ndarray, min_points: tuple, clusters: list, method_name: str) -> np.ndarray:
  rows_to_merge = prox_matrix[min_points,:]    # get the two rows to merge
  rows_to_merge[:,min_points] = 0.0            # zero out the columns where min_points lives
  # rows_to_merge are the two rows selected to be merged, we merge them with one of these methods
  # different update methods: min, max, average, centroids
  if method_name == "min":
    result = np.amin(rows_to_merge, axis=0)
  elif method_name == "max":
    result = np.amax(rows_to_merge, axis=0)
  elif method_name == "avg" or method_name == "average":
    num_points_i = len(clusters[min_points[0]])
    num_points_j = len(clusters[min_points[1]])
    cluster_i = rows_to_merge[0] * num_points_i
    cluster_j = rows_to_merge[1] * num_points_j

    result = (cluster_i + cluster_j) / (num_points_i + num_points_j)
  elif method_name == "cen" or method_name == "centroids":
    result = ((rows_to_merge[0] + rows_to_merge[1]) / 2)
  
  # print(f"\nAfter merging rows {min_points} together\n\t{result}\n")
  prox_matrix[ min_points[0] ] = result        # set row equal to result
  prox_matrix[ :,min_points[0] ] = result      # set column equal to result
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
        mergedMN[i] = cluster_children.index(clusters[point]) + cluster_length
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
def plot(linkage_matrix: np.ndarray, data:np.ndarray, title=None, color_map="RdBu", alpha=1.0, point_size=5, interactive=False):
  if interactive:
    plt.ion()
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

  titles_axes = ['Dendrogram (Click a point to set distance cut)', 'Cluster']
  titles_xy = [['Index', 'Distance'], ['X', 'Y']]
  fig.suptitle(title or 'Hierarchial Clustering')
  average_max = np.mean(linkage_matrix[-2:][:,2]) # this just computes the average value of the two greatest distances
  # just to initially split the dendrogram into two clusters
  dendro_line = axes[0].axhline(average_max, linewidth=0.3, color='r')
  scatter_labels = fcluster(linkage_matrix, average_max, criterion="distance")

  axes[1].scatter(data[:,0], data[:,1], s=point_size, c=scatter_labels, cmap=color_map, alpha=alpha)
  dendrogram(linkage_matrix,
          ax=axes[0],
          truncate_mode="lastp",
          p=10,
          orientation='top',)

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

def hierarchial(data, method_name):
  coords = data[:,:2]
  linkage_data_length = coords.shape[0] - 1
  distance_matrix = create_dist_matrix(coords) # distance matrix
  linkage_matrix = [] # the linkage data that explains how the hierarchial tree looks,
  clusters = [[i+1] for i in range(0,coords.shape[0])] # the individual points that we will let be individual clusters
  children = []

  for i in range(linkage_data_length):
    M, N, min_dist = find_min_node(distance_matrix)
    distance_matrix = update_dist_matrix(distance_matrix, ( M,N ), clusters, method_name)
    X, Y, num_children = merge(( M,N ), children, clusters, linkage_data_length + 1)
    linkage_matrix.append(( X, Y, min_dist, num_children ))

  linkage_matrix = np.array(linkage_matrix, dtype=np.double)
  # coordinates = pair(children)
  return linkage_matrix, coords

def iris_main():
  iris_data = np.loadtxt("./data/iris_data_set.csv", delimiter=",")
  iris_data = [ iris_data[ :,:2 ], iris_data[ :,2: ] ] # 3D array, [ Sepal Lengths [ [],[], ... ], Petal Lengths [ [], [], ... ]]
  
  for i,data in enumerate(iris_data):
    
    #### change second param to either min, max, avg, centroids #####
    linkage_matrix, coords = hierarchial(data, "max")
    # l = linkage(data, method="complete")
    plot(linkage_matrix, coords, title="Complete Linkage", color_map="Set1", point_size=10, interactive=True)

def hw_main():
  # x = [1, 2, 10, 5, 6, 4, 8, 0]
  # y = [3, 5, 4, 1, 2, 7, 3, 6]
  # coords = np.vstack((x, y)).transpose()
  data = np.loadtxt("./data/data11.txt", delimiter=",")

  #### change second param to either min, max, avg, centroids #####
  linkage_matrix, coords = hierarchial(data, "centroids")
  # l=linkage(data, method="complete")
  plot(linkage_matrix, coords, title="Complete Linkage", interactive=True)

## call either one to run hw or iris
hw_main()
# iris_main()