import matplotlib.pyplot as plt
from sklearn import datasets
from src import PCAPlottingExample

digits = datasets.load_digits()


def run():
    print("################################################################################")
    print("### VISUALIZING THE DATA SET")
    print("################################################################################")

    print("\n")
    print("### MatPlotLib Plots Ex.1:")
    print("- We will be creating a 10x10 grid displaying the images from digits.images and their corresponding answer from digits.target")
    print("- Look for the output to see the displayed figure")
    # Visualize the data with the matplotlib library
    # Create an image 6x6 inches
    figure = plt.figure(figsize=(6, 6))
    # Set the bounds of the plot
    for i in range(99):
        # 10x10 grid to place the image in
        ax = figure.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
        # Set the properties of the subplot
        ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
        # Place a text to show
        ax.text(0, 7, str(digits.target[i]))
        # Show the plot
    plt.show()

    print("\n")
    input("Press Enter to continue...")
    print("### MatPlotLib Plots Ex.2:")
    print("- We will be binding digits.images to digits.target to display the first 8 entries")
    print("- Look for the output to see the displayed figure")
    images_and_labels = list(zip(digits.images, digits.target))
    collection = enumerate(images_and_labels[:8]);
    for (index, (image, label)) in collection:
        # initialize a subplot of 2X4 at the (i+1)th position
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
    print("=> Next up, we will be visualizing the data by using Principle Component Analysis")
    print("\n")
    print("### END OF SECTION")
    print("/////////////////////////////////////////////////////////////////////////////////")
    print("\n")
    PCAPlottingExample.run()