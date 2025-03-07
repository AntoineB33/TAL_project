from II_Evaluation_forme_flechie import *


if __name__ == "__main__":
    args = sys.argv[1:]
    args = ["overwrite"]
    # args = ["no_new_test"]

    download_nltk_resources()
    prepare_all_data("III_Evaluation_lemme", *args)