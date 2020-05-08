import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

digits = datasets.load_digits()

example_index = 0

def run():
    print("################################################################################")
    print("### THE DIGITS DATA SET")
    print("################################################################################")

    # data contains 1797 examples of differently written digits as images
    # each image is a collection of 64 bits if the pixels that make up the images

    print("### INTRODUCTION:")
    print("- In this example we will be using the digits dataset included with the SciKit Learn library")
    print("- We will use the object called 'digits' to access the dataset")
    print("\n")

    print("### ABOUT THE DATASET:")
    print("- The dataset consists of 1797 examples of differently written digits as images")
    print("- The dataset contains three objects, 'data', 'target', and 'images. These are NumPy arrays")


    digits_data = digits.data
    digits_target = digits.target
    digits_images = digits.images
    print("\n")
    input("Press Enter to continue...")
    print("### 'DATA':")
    print("- digits_data shape: " + str(digits_data.shape))
    print("- digits_data at index " + str(example_index) +":")
    print(digits_data[example_index])
    print("=> 'Digits' is a 2D array. There are 1797 data entries, each composed of 64 features. These features are the individual pixels making up the image")


    print("\n")
    input("Press Enter to continue...")
    print("### 'TARGET':")
    print("- digits_target shape: " + str(digits_target.shape))
    print("- digits_target at index " + str(example_index) +":")
    print(digits_target[example_index])
    print("=> 'Target' is a 1D array. This array contains the 1797 correct answers to the 'data' that we will be using to train our model")
    print("=> The correct answer to the entry is " + str(digits_target[example_index]))

    print("\n")
    input("Press Enter to continue...")
    print("### 'IMAGES':")
    print("- digits_images shape: " + str(digits_images.shape))
    print("- digits_images at index " + str(example_index) + ":")
    print(digits_images[example_index])
    print("=> 'Images' is a 3D array. There are 1797 images corresponding to each data entry")
    print("=> Notice the shape of the digit represented above")

    print("\n")
    input("Press Enter to continue...")
    print("### PROPERTIES OF THE DATASET:")
    # if we get the array of unique elements from digits.target
    # we have an array of the 10 numbers in the data set
    print("- Since 'target' contains all possible answers, we can use NumPy to determine the number of unique answers")
    unique_digits = np.unique(digits.target)
    print("unique_digits: " + str(unique_digits))
    number_digits = len(unique_digits)
    print("number_digits: " + str(number_digits))

    print("\n")
    print("### END OF SECTION")
    print("/////////////////////////////////////////////////////////////////////////////////")
