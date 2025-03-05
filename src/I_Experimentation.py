import sys
import os

from prepare_data import *


if __name__ == "__main__":
    args = sys.argv[1:]

    lines = {}
    
    # Change to data directory
    os.chdir(Path("data/I_Experimentation/data_preparation/TRAIN_DEV_TEST_joints"))
    
    prepare_data(lines, "Europarl_train_10k", "", "Europarl_train_10k", tokenize=False, is_exo_3 = is_exo_3)
    prepare_data(lines, "Europarl_dev_1k", "", "Europarl_dev_1k", tokenize=False, is_exo_3 = is_exo_3)
    prepare_data(lines, "Europarl_test_500", "", "Europarl_test_500", tokenize=False, is_exo_3 = is_exo_3)

    # Change back to the original directory
    os.chdir(Path("../../../../.."))