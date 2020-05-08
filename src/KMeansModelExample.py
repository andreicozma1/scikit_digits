from sklearn import datasets, cluster, metrics
from sklearn.model_selection import train_test_split
import matplotlib.gridspec as gridspec
from sklearn.manifold import Isomap
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


digits = datasets.load_digits()
TEST_SIZE = 0.25
RANDOM_STATE = 40

def run():
    print("################################################################################")
    print("### KMEANS MODEL EXAMPLE")
    print("################################################################################")


    print("### INTRODUCTION:")
    print("- First, we use train_test_split to split our data into training and testing sets as described in the TrainTestSplitting tutorial")
    print("- We will be using two variables TEST_SIZE set to " + str(TEST_SIZE) + " and RANDOM_STATE set to " + str(RANDOM_STATE))
    # Create the Kmeans model
    # 10 digits for 10 clusters.
    # Use the same random_state variable as used before
    # Defaults to k-means++ so you can leave it out if you want
    # By adding the n-init argument to KMeans(), you can determine how many different centroid configurations the algorithm will try.
    X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(digits.data, digits.target,
                                                                                   digits.images, test_size=TEST_SIZE,
                                                                                   random_state=RANDOM_STATE)


    print("\n")
    input("Press Enter to continue...")
    print("### LOOKING AT OUR TRAINING DATA:")
    print("- We want to make sure that we are training our data on all the digits in the whole data set")
    unique_digits = np.unique(y_train)
    print("- unique_digits: " + str(unique_digits))
    number_digits = len(unique_digits)
    print("- n_digits len: " + str(number_digits))

    print("\n")
    input("Press Enter to continue...")
    print("### CREATING OUR KMEANS MODEL:")
    print("- We will use 'cluster' from sklearn to make a KMeans object with the following parameters")
    print("- Parameters: cluster.KMeans(init=\"k-means++\", n_clusters=number_digits, random_state=RANDOM_STATE)");
    clf = cluster.KMeans(init="k-means++", n_clusters=number_digits, random_state=RANDOM_STATE)
    print("- Printing out our model object:")
    print(clf)
    # fit the training data to the model

    print("\n")
    input("Press Enter to continue...")
    print("### TRAINING OUR MODEL")
    clf.fit(X_train)
    print("- Looking at our clusters:")
    print("- Cluster center shape: ")
    print(clf.cluster_centers_.shape)
    print("- Meaning: We have 10 cluster centers, each with 64 features coming from our pixel values")

    print("\n")
    input("Press Enter to continue...")
    print("### VISUALIZING OUR CLUSTER CENTERS:")
    print(" - We will use MatPlotLib to output the images for each of our cluster centers")
    print(" - Look for the output to see the displayed figure")

    # View the cluster center images on an 8x4 inch image
    fig = plt.figure(figsize=(8, 4))
    # Add title
    fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')
    # For all labels (0-9)
    for i in range(number_digits):
        # Initialize subplots in a grid of 2X5, at i+1th position
        ax = fig.add_subplot(2, 5, 1 + i)
        # Display images
        ax.imshow(clf.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary, interpolation='gaussian')
        # Don't show the axes
        plt.axis('off')
    # Show the plot
    plt.show()

    print("\n")
    input("Press Enter to continue...")
    print("### PREDICTING OUR TEST DATA:")
    print(" - Use the .predict function on the KMeans model to predict our X_test.")
    print(" - Ideally, these numbers should match our Y_test if the model is accurate")
    # Predict the labels of the test set
    y_pred = clf.predict(X_test)
    print(" - The first 100 predictions:")
    print(y_pred[:100])
    print(" - The first 100 expected results:")
    print(y_test[:100])
    print(" - As you can see, the KMeans model is not very accurate")

    print("\n")
    input("Press Enter to continue...")
    print("### EVALUATING OUR MODEL:")
    print("## CONFUSION MATRIX")
    print(" - We will use 'metrics' from sklearn to create a Confusion Matrix")
    print(" - More values in the diagonal means more accuracy")
    # Evaluation of clustering model
    # So far we can see that the model is incorect
    print(metrics.confusion_matrix(y_test, y_pred));

    input("Press Enter to continue...")
    print("## CLASSIFICATION REPORT")
    print(" - We can also create a Classification Report from the 'metrics' module")
    # Print the classification report of `y_test` and `predicted`
    print(metrics.classification_report(y_test, y_pred))
    # By looking at these numbers we can see that using the KMeans model is not
    # a good fit for our problem. This means that we must pick a better model for our data

    input("Press Enter to continue...")
    print("## OTHER STATISTICS")
    print('% 9s' % 'inertia    homo   compl  v-meas     ARI AMI  silhouette')
    print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (clf.inertia_,
             homogeneity_score(y_test, y_pred),
             completeness_score(y_test, y_pred),
             v_measure_score(y_test, y_pred),
             adjusted_rand_score(y_test, y_pred),
             adjusted_mutual_info_score(y_test, y_pred),
             silhouette_score(X_test, y_pred, metric='euclidean')))



    print("\n")
    input("Press Enter to continue...")
    print("### VISUALIZING USING PCA:")
    print("- Basic dimensionality reduction method")
    # Reduce the dimensions to visualize using PCA
    # Import `PCA()`
    # Model and fit the `digits` data to the PCA model
    print("- Create a PCA with 2 components, and fit our X data into it using 'fit_transform'")
    X_pca = PCA(n_components=2).fit_transform(X_train)
    # Compute cluster centers and predict cluster index for each sample
    print("- Predict some data from either the training or testing sets")
    clusters = clf.fit_predict(X_train)

    # Create a plot with subplots in a grid of 1X2
    print("- Use MatPlotLib to generate a plot of our actual vs predicted data")
    print("- Look for the output to see the displayed figure")
    fig = plt.figure(1, (8, 4))
    gs1 = gridspec.GridSpec(1, 2)
    ax = [fig.add_subplot(ss) for ss in gs1]
    # Adjust layout
    fig.suptitle('Predicted Versus Training Labels (PCA)', fontsize=14, fontweight='bold')
    # Add scatterplots to the subplots
    ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, edgecolors='black')
    ax[0].set_title('Predicted Training Labels')
    ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, edgecolors='black')
    ax[1].set_title('Actual Training Labels')
    gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    # Show the plots
    plt.show()



    print("\n")
    input("Press Enter to continue...")
    print("### VISUALIZING USING Isomap:")
    print("- Better dimensionality reduction method")
    # Reduce the dimensions to visualize using Isomap from SciKitLearn
    # Create an isomap and fit the `digits` data to it
    print("- Create a PCA with 2 components, and fit our X data into it using 'fit_transform'")
    X_iso = Isomap(n_neighbors=10).fit_transform(X_train)
    # Compute cluster centers and predict cluster index for each sample
    print("- Predict some data from either the training or testing sets")
    clusters = clf.fit_predict(X_train)

    print("- Use MatPlotLib to generate a plot of our actual vs predicted data")
    print("- Look for the output to see the displayed figure")
    # Create a plot with subplots in a grid of 1X2
    fig = plt.figure(1, (8, 4))
    gs1 = gridspec.GridSpec(1, 2)
    ax = [fig.add_subplot(ss) for ss in gs1]
    # Adjust layout
    fig.suptitle('Predicted Versus Training Labels (Isomap)', fontsize=14, fontweight='bold')
    # fig.subplots_adjust(top=0.85)
    # Add scatterplots to the subplots
    ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=clusters, edgecolors='black')
    ax[0].set_title('Predicted Training Labels')
    ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train, edgecolors='black')
    ax[1].set_title('Actual Training Labels')
    gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    # Show the plots
    plt.show()

    print("\n")
    print("### END OF SECTION")
    print("/////////////////////////////////////////////////////////////////////////////////")

