import sys
import os

from prepare_data import *


if __name__ == "__main__":
    args = sys.argv[1:]

    lines = {}
    
    # Change to data directory
    os.chdir(Path("data/I_Experimentation/data_preparation"))
    
    prepare_data(lines, "", "Europarl_train_10k", end = 10_000, toTokenize=False, is_exo_3 = False)
    prepare_data(lines, "", "Europarl_dev_1k", start = 10_000, length = 1_000, toTokenize=False, is_exo_3 = False)
    prepare_data(lines, "", "Europarl_test_500", start = 11_000, length = 500, toTokenize=False, is_exo_3 = False)

    # Change back to the original directory
    os.chdir(Path("../../../.."))