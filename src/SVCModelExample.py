from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
import matplotlib.gridspec as gridspec
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

digits = datasets.load_digits()
TEST_SIZE = 0.25
RANDOM_STATE = 40

def run():
    print("################################################################################")
    print("### SVC MODEL EXAMPLE")
    print("################################################################################")

    # Trying a different model
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(digits.data, digits.target,
                                                                                   digits.images, test_size=TEST_SIZE,
                                                                                   random_state=RANDOM_STATE)
    print("### INTRODUCTION:")
    print("- First, we use train_test_split to split our data into training and testing sets as described in the TrainTestSplitting tutorial")
    print("- We will be using two variables TEST_SIZE set to " + str(TEST_SIZE) + " and RANDOM_STATE set to " + str(RANDOM_STATE))

    print("\n")
    print("### LOOKING AT OUR TRAINING DATA:")
    print("- We want to make sure that we are training our data on all the digits in the whole data set")
    unique_digits = np.unique(y_train)
    print("unique_digits: " + str(unique_digits))
    number_digits = len(unique_digits)
    print("n_digits len: " + str(number_digits))

    print("\n")
    print("### CREATING OUR SVC MODEL:")
    print("- We will use 'svm' from sklearn to make an SVC Model with the following parameters")
    print("- Parameters: svm.SVC(gamma=100, C=100, kernel='linear')");
    # Create the SVC model
    svc_model = svm.SVC(gamma=100, C=100, kernel='linear')

    print("- Fit the X_train and Y_train data to our model")
    svc_model.fit(X_train, y_train)
    print("- SVC Score for test data: " + str(svc_model.score(X_test, y_test)))
    print("- As we can see, this is much better than the KMeans approach")
    print("- Can we do even better?")

    print("\n")
    print("### USING GRIDSEARCH TO FIND GOOD PARAMETERS FOR OUR MODEL:")
    print(" - We will use 'GridSearchCV' from sklearn.model_selection to accomplish this")
    print(" - We will use different key value pairs for possible parameters of Gamma and C to test out for our model")
    # Try to find the best parameters for the model
    # Set the parameter candidates
    parameter_candidates = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    print("- Create a GridSearchCV object with the following parameters")
    print("- Parameters: clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)")
    # Create a classifier with the parameter candidates
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
    # Train the classifier on training data
    print("- Fit the training data to our classifier")
    clf.fit(X_train, y_train)
    print("- The results are as follows:")
    # Print out the results
    print('=> Best score for training data:', clf.best_score_)
    print('=> Best `C`:', clf.best_estimator_.C)
    print('=> Best kernel:', clf.best_estimator_.kernel)
    print('=> Best `gamma`:', clf.best_estimator_.gamma)
    # After finding the best parameters to use,
    # Apply the classifier to the test data, and view the accuracy score
    print("- SVC Score for test data from classifier: " + str(clf.score(X_test, y_test)))
    # Train and score a new classifier with the grid search parameters


    print("\n")
    print("### RECREATE THE SVC MODEL WITH THE OPTIMAL PARAMETERS WE FOUND:")
    svc_model = svm.SVC(C=clf.best_estimator_.C, kernel=clf.best_estimator_.kernel, gamma=clf.best_estimator_.gamma)
    svc_model.fit(X_train, y_train)
    print("-> SVC Score with optimal parameters: " + str(svc_model.score(X_test, y_test)))
    print("\n")
    print("### PREDICTING OUR TEST DATA:")
    print(" - Use the .predict function on the SVC model to predict our X_test.")
    print(" - Ideally, these numbers should match our Y_test if the model is accurate")
    y_pred = svc_model.predict(X_test)
    print(" - The first 100 predictions:")
    print(y_pred[:100])
    print(" - The first 100 expected results:")
    print(y_test[:100])
    print(" - As you can see, the SVC model is way more accurate")



    print("\n")
    print("### EVALUATING OUR MODEL:")
    print("## CONFUSION MATRIX")
    print(" - We will use 'metrics' from sklearn to create a Confusion Matrix")
    print(" - More values in the diagonal means more accuracy")
    # Evaluation of clustering model
    # So far we can see that the model is incorect
    print(metrics.confusion_matrix(y_test, y_pred));

    print("## CLASSIFICATION REPORT")
    print(" - We can also create a Classification Report from the 'metrics' module")
    # Print the classification report of `y_test` and `predicted`
    print(metrics.classification_report(y_test, y_pred))
    # By looking at these numbers we can see that using the KMeans model is not
    # a good fit for our problem. This means that we must pick a better model for our data

    print("## OTHER STATISTICS")
    print('homo    compl   v-meas     ARI     AMI  silhouette')
    print('%.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (homogeneity_score(y_test, y_pred),
             completeness_score(y_test, y_pred),
             v_measure_score(y_test, y_pred),
             adjusted_rand_score(y_test, y_pred),
             adjusted_mutual_info_score(y_test, y_pred),
             silhouette_score(X_test, y_pred, metric='euclidean')))



    print("### CREATING AN IMAGE WITH SAMPLE PREDICTED VALUES AND THEIR IMAGES:")
    print("- We will be using MatPlotLib")
    print("- Look for the output to see the displayed figure")
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



    print("\n")
    print("### VISUALIZING USING PCA:")
    print("- Basic dimensionality reduction method")
    # Reduce the dimensions to visualize using PCA
    # Import `PCA()`
    # Model and fit the `digits` data to the PCA model
    print("- Create a PCA with 2 components, and fit our X data into it using 'fit_transform'")
    X_pca = PCA(n_components=2).fit_transform(X_train)
    # Compute cluster centers and predict cluster index for each sample
    print("- Predict some data from either the training or testing sets")
    predicted = svc_model.predict(X_train)

    # Create a plot with subplots in a grid of 1X2
    print("- Use MatPlotLib to generate a plot of our actual vs predicted data")
    print("- Look for the output to see the displayed figure")
    fig = plt.figure(1, (8, 4))
    gs1 = gridspec.GridSpec(1, 2)
    ax = [fig.add_subplot(ss) for ss in gs1]
    # Adjust layout
    fig.suptitle('Predicted Versus Training Labels (PCA)', fontsize=14, fontweight='bold')
    # Add scatterplots to the subplots
    ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=predicted, edgecolors='black')
    ax[0].set_title('Predicted Training Labels')
    ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, edgecolors='black')
    ax[1].set_title('Actual Training Labels')
    gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    # Show the plots
    plt.show()



    print("\n")
    print("### VISUALIZING USING Isomap:")
    print("- Better dimensionality reduction method")
    # Reduce the dimensions to visualize using Isomap from SciKitLearn
    # Create an isomap and fit the `digits` data to it
    print("- Create a PCA with 2 components, and fit our X data into it using 'fit_transform'")
    X_iso = Isomap(n_neighbors=10).fit_transform(X_train)
    # Compute cluster centers and predict cluster index for each sample
    print("- Predict some data from either the training or testing sets")
    predicted = svc_model.predict(X_train)

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
    ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=predicted, edgecolors='black')
    ax[0].set_title('Predicted Training Labels')
    ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train, edgecolors='black')
    ax[1].set_title('Actual Training Labels')
    gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    # Show the plots
    plt.show()

    print("\n")
    print("### END OF SECTION")
    print("/////////////////////////////////////////////////////////////////////////////////")