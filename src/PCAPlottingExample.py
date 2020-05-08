from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

digits = datasets.load_digits()

def run():
    print("################################################################################")
    print("### PRINCIPLE COMPONENT ANALYSIS")
    print("################################################################################")
    print("### INTRODUCTION:")
    print("- We will be using the PCA from sklearn.decomposition to be able to visualize our data on a scatter plot")
    print("- To do so, we need to only keep 2 components, since we want to visualize on a 2D plane")

    # Visualization - PCA (two components to have two dimensional data to plot)
    print("\n")
    input("Press Enter to continue...")
    print("### CREATING THE PCA:")
    pca = PCA(n_components=2)
    print("- This is the 'pca' object printed out:")
    print(pca)
    reduced_data_pca = pca.fit_transform(digits.data)
    print("- Run pca.fit_transform on digits.data")
    print("- The result is data narrowed down to 2D which we can put on a scatter plot")
    print(reduced_data_pca)

    print("\n")
    input("Press Enter to continue...")
    print("### CREATING THE SCATTER PLOT WITH MatPlotLib:")
    print("- Knowing the number of total unique digits in our digits.target, we should have the same number differently colored clusters on our scatterplot")
    print("- Look for the output to see the displayed figure")
    # Assign a color for each of the 10 unique answers for our data set
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        x = reduced_data_pca[:, 0][digits.target == i]
        y = reduced_data_pca[:, 1][digits.target == i]
        plt.scatter(x, y, c=colors[i], edgecolors='black')
    # Add a legend
    plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # Add labels
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    # Add a title
    plt.title("PCA Scatter Plot")
    plt.show()

    print("\n")
    print("### END OF SECTION")
    print("/////////////////////////////////////////////////////////////////////////////////")