import sys
import os

from prepare_data import *


if __name__ == "__main__":
    args = sys.argv[1:]

    lines = {}
    
    # Change to data directory
    os.chdir(Path("data/I_Experimentation/data_preparation"))
    
    prepare_data(lines, "../TRAIN_data", "Europarl_train_10k", toTokenize=False, inputPath="Europarl_train_10k")
    prepare_data(lines, "../DEV_data", "Europarl_dev_1k", toTokenize=False, inputPath="Europarl_dev_1k")
    prepare_data(lines, "../TEST_data", "Europarl_test_500", toTokenize=False, inputPath="Europarl_test_500")

    # Change back to the original directory
    os.chdir(Path("../../../.."))