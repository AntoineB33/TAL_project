import os
import subprocess
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

# Set Moses home directory (relative to original script location)
MOSES_HOME = "../../../../../../Training/SMT"

# Change to data directory
data_dir = Path("data/II_Evaluation_forme_flechie/data_preparation/Europarl.en-fr.txt")
os.chdir(data_dir)

def getLines(lines, input_path):
    if not lines[input_path]:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines[input_path] = f.readlines()

# Utility function for tokenization
def tokenize(input_file, output_file, lang):
    tokenizer = Path(MOSES_HOME) / "mosesdecoder/scripts/tokenizer/tokenizer.perl"
    cmd = ["perl", str(tokenizer), "-l", lang]
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        subprocess.run(cmd, stdin=infile, stdout=outfile, check=True)

# Utility functions for truecasing
def train_truecaser(corpus_path, model_path):
    trainer = Path(MOSES_HOME) / "mosesdecoder/scripts/recaser/train-truecaser.perl"
    cmd = ["perl", str(trainer), "--model", model_path, "--corpus", corpus_path]
    subprocess.run(cmd, check=True)

def apply_truecaser(model_path, input_path, output_path):
    truecaser = Path(MOSES_HOME) / "mosesdecoder/scripts/recaser/truecase.perl"
    cmd = ["perl", str(truecaser), "--model", model_path]
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        subprocess.run(cmd, stdin=infile, stdout=outfile, check=True)

# d. Cleaning corpus
def clean_corpus(base_name, lang1, lang2, output_base, min_len, max_len):
    cleaner = Path(MOSES_HOME) / "mosesdecoder/scripts/training/clean-corpus-n.perl"
    cmd = ["perl", str(cleaner), base_name, lang1, lang2, output_base, 
           str(min_len), str(max_len)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    

    args = sys.argv[1:]
    lines = {"Europarl.en-fr.en": [],
             "Europarl.en-fr.fr": [],
             "EMEA.en-fr.en": [],
             "EMEA.en-fr.fr": [],
             "../../TEST_data/Europarl_out_domain.tok.true.clean.en": [],
             "../../TEST_data/Europarl_out_domain.tok.true.clean.fr": []}

    # check if the data has already been processed
    if not Path("../../TRAIN_data/Europarl_train_100k.tok.true.clean.en").exists():
        # a. Splitting data
        print("Splitting data...")
        getLines(lines, "Europarl.en-fr.en")
        with open("Europarl_train_100k.en", 'w', encoding='utf-8') as f:
            f.writelines(lines["Europarl.en-fr.en"][:100_000])

        # b. Tokenization
        print("Tokenizing...")
        tokenize("Europarl_train_100k.en", "Europarl_train_100k.tok.en", "en")
        tokenize("Europarl_train_100k.fr", "Europarl_train_100k.tok.fr", "fr")

        # c. Truecasing
        print("Truecasing...")
        # c.1 Training truecase models
        train_truecaser("Europarl_train_100k.tok.en", "truecase-model_train.en")
        train_truecaser("Europarl_train_100k.tok.fr", "truecase-model_train.fr")

        # c.2 Applying truecasing
        apply_truecaser("truecase-model_train.en", "Europarl_train_100k.tok.en", 
                        "Europarl_train_100k.tok.true.en")
        apply_truecaser("truecase-model_train.fr", "Europarl_train_100k.tok.fr", 
                        "Europarl_train_100k.tok.true.fr")
        
        # d. Cleaning corpus
        print("Cleaning corpus...")
        clean_corpus("Europarl_train_100k.tok.true", "fr", "en", 
                    "../../TRAIN_data/Europarl_train_100k.tok.true.clean", 1, 80)

    if not Path("../../DEV_data/Europarl_dev_3750.tok.true.clean.en").exists():
        # a. Splitting data
        print("Splitting data...")
        getLines(lines, "Europarl.en-fr.fr")
        with open("Europarl_dev_3750.fr", 'w', encoding='utf-8') as f:
            f.writelines(lines["Europarl.en-fr.fr"][100_000:103_750])

        # b. Tokenization
        print("Tokenizing...")
        tokenize("Europarl_dev_3750.en", "Europarl_dev_3750.tok.en", "en")
        tokenize("Europarl_dev_3750.fr", "Europarl_dev_3750.tok.fr", "fr")

        # c. Truecasing
        print("Truecasing...")
        # c.1 Training truecase models
        train_truecaser("Europarl_dev_3750.tok.en", "truecase-model_dev.en")
        train_truecaser("Europarl_dev_3750.tok.fr", "truecase-model_dev.fr")

        # c.2 Applying truecasing
        apply_truecaser("truecase-model_dev.en", "Europarl_dev_3750.tok.en", 
                        "Europarl_dev_3750.tok.true.en")
        apply_truecaser("truecase-model_dev.fr", "Europarl_dev_3750.tok.fr", 
                        "Europarl_dev_3750.tok.true.fr")

        # d. Cleaning corpus
        print("Cleaning corpus...")
        clean_corpus("Europarl_dev_3750.tok.true", "fr", "en", 
                    "../../DEV_data/Europarl_dev_3750.tok.true.clean", 1, 80)
    
    # for the test, getting random pairs of lines inside the domain using train_test_split
    getLines(lines, "Europarl.en-fr.en")
    getLines(lines, "Europarl.en-fr.fr")
    X_train, X_test, y_train, y_test = train_test_split(lines["Europarl.en-fr.en"], lines["Europarl.en-fr.fr"], test_size=500)

    



    print("Data preparation complete.")