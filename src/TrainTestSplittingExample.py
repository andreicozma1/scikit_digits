from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

digits = datasets.load_digits()
TEST_SIZE = 0.25
RANDOM_STATE = 40

def run():
    print("################################################################################")
    print("### SPLITTING THE DATA INTO TRAINING AND TESTING SETS")
    print("################################################################################")


    print("### INTRODUCTION:")
    print("- We will be using 'train_test_split' from 'sklearn.model_selection' to split our data")
    print("- 'train_test_split' uses digits.data, digits.target and digits.images to split them into separate training and testing sets")
    print("- We will be using two variables TEST_SIZE set to " + str(TEST_SIZE) + " and RANDOM_STATE set to " + str(RANDOM_STATE))
    print("- This will give us " + str(TEST_SIZE * 100) + "% test data and " + str(100 - TEST_SIZE * 100) + "% training data for our model")
    print("- The RANDOM_STATE is simply a seed used for randomly splitting our data")
    # Splitting the data into training and testing sets
    # to visualize y = mx + b with y being the answer
    # in this case, the y_train is the set containing the target value of the corresponding data
    X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(digits.data, digits.target,
                                                                                   digits.images, test_size=TEST_SIZE,
                                                                                   random_state=RANDOM_STATE)

    print("\n")
    input("Press Enter to continue...")
    print("### PROPERTIES OF OUR SPLIT DATA:")
    n_samples, n_features = X_train.shape
    # 3/4ths of the data is used to train
    # 1/4th of the data is used to test
    print("- Looking at the shape of our data, we can see the number of elements in each set")
    print("=> X_train:")
    print("- samples: " + str(n_samples))
    # 64 different features because there are 64 pixels in each image
    print("- n_features: " + str(n_features))

    n_samples, n_features = X_test.shape
    print("=> X_test:")
    print("- samples: " + str(n_samples))
    # 64 different features because there are 64 pixels in each image
    print("- n_features: " + str(n_features))


    print("\n")
    input("Press Enter to continue...")
    print("### LOOKING AT OUR TRAINING DATA")
    print("- We want to make sure that we are training our data on all the digits in the whole data set")
    unique_digits = np.unique(y_train)
    print("unique_digits: " + str(unique_digits))
    number_digits = len(unique_digits)
    print("n_digits len: " + str(number_digits))

    print("\n")
    print("### END OF SECTION")
    print("/////////////////////////////////////////////////////////////////////////////////")
