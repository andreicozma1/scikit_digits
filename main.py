from src import DataSets
from src import Visualization
from src import TrainTestSplittingExample
from src import SVCModelExample
from src import KMeansModelExample


def main():
    DataSets.run()
    print("\n")
    Visualization.run()
    print("\n")
    TrainTestSplittingExample.run()
    print("\n")
    KMeansModelExample.run()
    print("\n")
    SVCModelExample.run()


main()


