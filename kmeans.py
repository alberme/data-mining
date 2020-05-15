import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=True)

def compute_distances(samples, centroids):
    # so you can send samples param without slicing the labels off first
    sliced_centroids = centroids[ :,:2 ] if centroids.shape[1] > 1 else centroids

    distances = np.zeros((centroids.shape[0], samples.shape[0], 2), dtype=np.float32)
    for i,centroid in enumerate(sliced_centroids):
        for j,sample in enumerate(samples):
            distances[i,j] = (sample - centroid)**2
    
    distances = np.sum(distances, axis=2)
    distances = np.sqrt(distances)

    return distances

def compute_centroids(samples, samples_labels, centroids) -> np.ndarray:
    # for each centroid, compute the mean of all the samples, result is centroids new location
    recomputed_centroids = []

    for k in range(centroids.shape[0]):
        new_centroid = samples[ samples_labels == k+1 ].mean(axis=0)
        recomputed_centroids.append( np.hstack(( new_centroid, [k+1] )) )
    # recomputed_centroids = [ samples[ samples_labels == k+1 ].mean(axis=0) for k in range(centroids.shape[0]) ]

    return np.array(recomputed_centroids, dtype=np.float32)

def init_centroids(samples: np.ndarray, num_clusters: int) -> np.ndarray:
    shuffled_samples = np.copy(samples)    # copy the sample set first
    np.random.shuffle(shuffled_samples)    # then card shuffle the data set
    # grab every k'th sample, or row, from the shuffled samples, then append the value k on the end of the row we grabbed
    centroids = [ np.hstack( (shuffled_samples[k], [k+1]) ) for k in range(num_clusters) ]

    return np.array(centroids, dtype=np.float32)

def kmeans(samples: np.ndarray, k: int) -> tuple:
    i = 0   # kmeans iteration count just to keep track 
    # we only want the first two columns of the data set if it happens to be more than 2 columns wide
    if samples.shape[1] > 2:
        samples = samples[ :,:2 ]

    centroids = init_centroids(samples, k)
    prev_centroids = np.empty((k,3), dtype=np.float32)
    samples_labels = np.zeros((samples.shape[0],)) # the labels computed from argmin()

    while not np.all(centroids == prev_centroids):
        prev_centroids = centroids.copy()
        distances = compute_distances(samples, centroids)
        samples_labels = np.argmin(distances, axis=0) + 1
        centroids = compute_centroids(samples, samples_labels, centroids)
        i += 1

    return (samples_labels, centroids, i)

def karate_main():
    k = 2
    color_map = "winter"
    karate_data = np.loadtext("./data/KarateMatrix.txt")

def iris_main():
    k = 3 # num clusters, you can change this! Just dont go too high!
    color_map = "winter"
    iris_data = np.loadtxt("./data/iris_data_set.csv", delimiter=",")
    iris_data = [ iris_data[ :,:2 ], iris_data[ :,2: ] ] # 3D array, [ Sepal Lengths [ [],[], ... ], Petal Lengths [ [], [], ... ]]
    iris_plots_titles = ["Sepal", "Petal"]
    iris_axes_titles = [["Sepal Length (cm)", "Sepal Width (cm)"], ["Petal Length (cm)", "Petal Width (cm)"]]


    fig, plots = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    for i,data in enumerate(iris_data):
        labels, centroids, kmeans_i = kmeans(data, k)
        x_name, y_name = iris_axes_titles[i]

        print(f"Number of kmeans iterations performed for {iris_plots_titles[i]}: {kmeans_i}")
        plots[i].set_title(iris_plots_titles[i])
        plots[i].scatter(data[:,0], data[:,1], c=labels, cmap=color_map, alpha=0.5)
        plots[i].scatter(centroids[:,0], centroids[:,1], c="r")
        plots[i].set(xlabel=x_name, ylabel=y_name)

    fig.suptitle('Iris Data Set')
    fig.tight_layout()
    plt.show()

def hw_main():
    k = 3 # num clusters, you can change this! Just dont go too high!
    color_map = "RdBu" if k <= 2 else "spring"
    data = np.loadtxt("./data/data11.txt", delimiter=",")

    labels, centroids, i = kmeans(data, k)

    print(f"Number of kmeans iterations performed: {i}")
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:,0], data[:,1], c=labels, cmap=color_map, alpha=0.5)
    plt.scatter(centroids[:,0], centroids[:,1], c="r")
    plt.show()

#### call either iris_main or hw_main, or both! 😂😂
iris_main()