from sklearn import datasets, cluster, metrics, svm
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.manifold import Isomap
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score

# digits_train = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None)
# digits_test = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes", header=None)
#
# print(digits_train)
# print(digits_test)

TEST_SIZE = 0.25
RANDOM_STATE = 40
GAMMA = 100

digits = datasets.load_digits()

# 4 dimensional array
# (2,3,4,5) is a tuple.
# np.zeros takes in a tuple
# y = np.zeros((2,3,4,5))
# print(y)
# print(y.shape)

# data contains 1797 examples of differently written digits as images
# each image is a collection of 64 bits if the pixels that make up the images
digits_data = digits.data
print("digits_data shape: " + str(digits_data.shape))
print("\n")
# target contains the correct prediction to be used to train the data
digits_target = digits.target
print("digits_target shape: " + str(digits_target.shape))
print("\n")

# if we get the array of unique elements from digits.target
# we have an array of the 10 numbers in the data set
unique_digits = np.unique(digits.target)
print("unique_digits: " + str(unique_digits))
number_digits = len(unique_digits)
print("number_digits: " + str(number_digits))
print("\n")

digits_images = digits.images
# the first example in the training data set is 0, which looks like this as an array of pixel values
print("First example image:")
print(digits_images[0])
# the correct answer to this is below (0)
print("Answer: " + str(digits_target[0]))


# Visualize the data with the matplotlib library
# Create an image 6x6 inches
figure = plt.figure(figsize=(6, 6))
# Set the bounds of the plot
# figure.subplots_adjust(left=0, right=1,bottom=0,top=1,hspace=0.05, wspace=0.05)

for i in range(99):
    # 8x8 grid to place the image in
    ax = figure.add_subplot(10,10, i + 1, xticks=[], yticks=[])
    # Set the properties of the subplot
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # Place a text to show
    ax.text(0,7, str(digits.target[i]))
# Show the plot
plt.show()


images_and_labels = list(zip(digits.images, digits.target))
collection = enumerate(images_and_labels[:8]);
for (index, (image, label)) in collection:
    # initialize a subplot of 2X4 at the i+1-th position
    plt.subplot(2, 4, index + 1)
    # Don't plot any axes
    plt.axis('off')
    # Display images in all subplots
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    # Add a title to each subplot
    plt.title('Training: ' + str(label))
# Show the plot
plt.show()


print("\n")
print("Create PCA:")
# Visualization - PCA (two components to have two dimensional data to plot)
# Randomized PCA seems to perform better with high dimension data
pca = PCA(n_components=2)
reduced_data_pca = pca.fit_transform(digits.data)
print(reduced_data_pca)

colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
    x = reduced_data_pca[:, 0][digits.target == i]
    y = reduced_data_pca[:, 1][digits.target == i]
    plt.scatter(x, y, c=colors[i], edgecolors='black')
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
plt.show()


print("\n")
print("Pre-Processing the Data:")
# Pre-processing your data
# By scaling the data, you shift the distribution of each attribute to have a mean of zero and a standard deviation of one (unit variance).
data_standardized = scale(digits.data)

# Splitting the data into training and testing sets
# to visualize y = mx + b with y being the answer
# in this case, the y_train is the set containing the target value of the corresponding data
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(digits_data, digits.target, digits.images, test_size=TEST_SIZE, random_state=RANDOM_STATE)

n_samples, n_features = X_train.shape
# 3/4ths of the data is used to train
# 1/4th of the data is used to test
print("n_samples: " + str(n_samples))
# 64 different features because there are 64 pixels in each image
print("n_features: " + str(n_features))

n_digits = len(np.unique(y_train))
print("n_digits len: " + str(len(y_train)))
print("n_digits: " + str(n_digits))

# Create the Kmeans model
# 10 digits for 10 clusters.
# Use the same random_state variable as used before
# Defaults to k-means++ so you can leave it out if you want
# By adding the n-init argument to KMeans(), you can determine how many different centroid configurations the algorithm will try.
clf = cluster.KMeans(init="k-means++", n_clusters=n_digits,random_state=RANDOM_STATE)
# fit the training data to the model
clf.fit(X_train)


# View the cluster center images
fig = plt.figure(figsize=(8, 4))
# Add title
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')
# For all labels (0-9)
for i in range(10):
    # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)
    # Display images
    ax.imshow(clf.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary, interpolation='gaussian')
    # Don't show the axes
    plt.axis('off')
# Show the plot
plt.show()

# Predict the labels of the test set
y_pred = clf.predict(X_test)
print(y_pred[:100])
print(y_test[:100])
# Study the shape of the cluster centers.
print(clf.cluster_centers_.shape)


# Evaluation of clustering model
# So far we can see that the model is incorect
print(metrics.confusion_matrix(y_test,y_pred));
# Print the classification report of `y_test` and `predicted`
print(metrics.classification_report(y_test, y_pred))
# By looking at these numbers we can see that using the KMeans model is not
# a good fit for our problem. This means that we must pick a better model for our data
print('% 9s' % 'inertia    homo   compl  v-meas     ARI AMI  silhouette')
print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          %(clf.inertia_,
      homogeneity_score(y_test, y_pred),
      completeness_score(y_test, y_pred),
      v_measure_score(y_test, y_pred),
      adjusted_rand_score(y_test, y_pred),
      adjusted_mutual_info_score(y_test, y_pred),
      silhouette_score(X_test, y_pred, metric='euclidean')))
print("\n")



# Reduce the dimensions to visualize using Isomap from SciKitLearn
# Create an isomap and fit the `digits` data to it
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)
# Compute cluster centers and predict cluster index for each sample
clusters = clf.fit_predict(X_train)
# Create a plot with subplots in a grid of 1X2
fig = plt.figure(1,(8,4))
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

# Reduce the dimensions to visualize using PCA
# Import `PCA()`
from sklearn.decomposition import PCA
# Model and fit the `digits` data to the PCA model
X_pca = PCA(n_components=2).fit_transform(X_train)
# Compute cluster centers and predict cluster index for each sample
clusters = clf.fit_predict(X_train)
# Create a plot with subplots in a grid of 1X2
fig = plt.figure(1,(8,4))
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






# Trying a different model
# Split the data into training and test sets
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(digits.data, digits.target, digits.images, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Create the SVC model
svc_model = svm.SVC(gamma=GAMMA, C=100, kernel='linear')

# Try to find the best parameters for the model
# Set the parameter candidates
parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]
# Create a classifier with the parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
# Train the classifier on training data
clf.fit(X_train, y_train)
# Print out the results
print('Best score for training data:', clf.best_score_)
print('Best `C`:',clf.best_estimator_.C)
print('Best kernel:',clf.best_estimator_.kernel)
print('Best `gamma`:',clf.best_estimator_.gamma)

print("\n")
# After finding the best parameters to use,
# Apply the classifier to the test data, and view the accuracy score
print("SVC Score: " + str(clf.score(X_test, y_test)))
# Train and score a new classifier with the grid search parameters

svc_model = svm.SVC(C=clf.best_estimator_.C, kernel=clf.best_estimator_.kernel, gamma=clf.best_estimator_.gamma)
svc_model.fit(X_train, y_train)
print("SVC Score with optimal parameters: " + str(svc_model.score(X_test, y_test)))

y_pred = svc_model.predict(X_test)
print(y_pred[:100])
print(y_test[:100])


# Zip together the `images_test` and `predicted` values in `images_and_predictions`
images_and_predictions = list(zip(images_test, y_pred))

# For the first 4 elements in `images_and_predictions`
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    # Initialize subplots in a grid of 1 by 4 at positions i+1
    plt.subplot(1, 4, index + 1)
    # Don't show axes
    plt.axis('off')
    # Display images in all subplots in the grid
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    # Add a title to the plot
    plt.title('Predicted: ' + str(prediction))
# Show the plot
plt.show()

# See how our model is performing
# Evaluation of clustering model
# Print the confusion matrix of `y_test` and `predicted`
print(metrics.confusion_matrix(y_test, y_pred))
# Print the classification report of `y_test` and `predicted`
print(metrics.classification_report(y_test, y_pred))



# Create an isomap and fit the `digits` data to it
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)
# Compute cluster centers and predict cluster index for each sample
predicted = svc_model.predict(X_train)
# Create a plot with subplots in a grid of 1X2
fig = plt.figure(1,(8,4))
gs1 = gridspec.GridSpec(1, 2)
ax = [fig.add_subplot(ss) for ss in gs1]
# Adjust the layout
fig.subplots_adjust(top=0.85)
# Add scatterplots to the subplots
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=predicted, edgecolors='black')
ax[0].set_title('Predicted labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train, edgecolors='black')
ax[1].set_title('Actual Labels')
# Add title
fig.suptitle('Predicted versus actual labels', fontsize=14, fontweight='bold')
gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
# Show the plot
plt.show()