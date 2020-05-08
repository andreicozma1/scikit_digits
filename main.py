from src import DataSets
from src import Visualization
from src import TrainTestSplittingExample
from src import SVCModelExample
from src import KMeansModelExample


def main():
    DataSets.run()
    print("\n")
    raw_input("Press Enter to continue...")
    Visualization.run()
    print("\n")
    raw_input("Press Enter to continue...")
    TrainTestSplittingExample.run()
    print("\n")
    raw_input("Press Enter to continue...")
    KMeansModelExample.run()
    print("\n")
    raw_input("Press Enter to continue...")
    SVCModelExample.run()


main()


