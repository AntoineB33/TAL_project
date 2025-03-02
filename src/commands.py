import os
import subprocess
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

WSL_PATH = "/mnt/c/Users/abarb/Documents/travail/Polytech Paris Saclay/cours/et5/TAL/projet/git/TAL_project"

# Set Moses home directory (relative to the project location)
MOSES_HOME = "../../../../../../Training/SMT"


class WSLSession:
    def __init__(self):
        # Start a persistent WSL bash session
        self.process = subprocess.Popen(["wsl"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        # Change to the project directory
        self.run_command(f"cd {WSL_PATH}")

    def run_command(self, command):
        print(f"Running: {command}")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def close(self):
        # Exit the bash session
        self.process.stdin.write("exit\n")
        self.process.stdin.flush()
        self.process.wait()

def getLines(lines, input_path):
    if not lines[input_path]:
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                lines[input_path] = f.readlines()
        except FileNotFoundError:
            print(f"File {input_path} not found.")
            sys.exit(1)

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

    # `lines` stores the lines of the different files
    lines = {"Europarl.en-fr.en": [],
             "Europarl.en-fr.fr": [],
             "EMEA.en-fr.en": [],
             "EMEA.en-fr.fr": [],
             "../../TEST_data/Europarl_out_domain.tok.true.clean.en": [],
             "../../TEST_data/Europarl_out_domain.tok.true.clean.fr": []}


    # Change to data directory
    os.chdir(Path("data/II_Evaluation_forme_flechie/data_preparation/Europarl.en-fr.txt"))
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
    _, _, test_en_lines, test_fr_lines = train_test_split(lines["Europarl.en-fr.en"][:103_750], lines["Europarl.en-fr.fr"][:103_750], test_size=500)
    with open("../../TEST_data/Europarl_in_domain.tok.true.clean.en", 'w', encoding='utf-8') as f:
        f.writelines(test_en_lines)
    with open("../../TEST_data/Europarl_in_domain.tok.true.clean.fr", 'w', encoding='utf-8') as f:
        f.writelines(test_fr_lines)

    # for the test, getting random pairs of lines outside the domain using train_test_split
    _, _, test_en_lines, test_fr_lines = train_test_split(lines["Europarl.en-fr.en"][103_750:], lines["Europarl.en-fr.fr"][103_750:], test_size=500)
    with open("../../TEST_data/Europarl_out_domain.tok.true.clean.en", 'w', encoding='utf-8') as f:
        f.writelines(test_en_lines)
    with open("../../TEST_data/Europarl_out_domain.tok.true.clean.fr", 'w', encoding='utf-8') as f:
        f.writelines(test_fr_lines)

    # Change to data directory
    os.chdir(Path("../EMEA.en-fr.txt"))
    getLines(lines, "EMEA.en-fr.en")
    getLines(lines, "EMEA.en-fr.fr")
    if not Path("../../TRAIN_data/Europarl_EMEA_train_100k_10k.tok.true.clean.en").exists():
        # a. Splitting data
        print("Splitting data...")
        with open("EMEA_train_10k.en", 'w', encoding='utf-8') as f:
            f.writelines(lines["EMEA.en-fr.en"][:10_000])
        with open("EMEA_train_10k.fr", 'w', encoding='utf-8') as f:
            f.writelines(lines["EMEA.en-fr.fr"][:10_000])
        
        # b. Tokenization
        print("Tokenizing...")
        tokenize("EMEA_train_10k.en", "EMEA_train_10k.tok.en", "en")
        tokenize("EMEA_train_10k.fr", "EMEA_train_10k.tok.fr", "fr")

        # c. Truecasing
        print("Truecasing...")
        # c.1 Training truecase models
        train_truecaser("EMEA_train_10k.tok.en", "truecase-model_train.en")
        train_truecaser("EMEA_train_10k.tok.fr", "truecase-model_train.fr")

        # c.2 Applying truecasing
        apply_truecaser("truecase-model_train.en", "EMEA_train_10k.tok.en", 
                        "EMEA_train_10k.tok.true.en")
        apply_truecaser("truecase-model_train.fr", "EMEA_train_10k.tok.fr",
                        "EMEA_train_10k.tok.true.fr")
        
        # d. Cleaning corpus
        print("Cleaning corpus...")
        clean_corpus("EMEA_train_10k.tok.true", "fr", "en", 
                    "../../TRAIN_data/EMEA_train_10k.tok.true.clean", 1, 80)
        
        # e. Combining corpora
        print("Combining corpora...")
        with open("../../TRAIN_data/Europarl_EMEA_train_100k_10k.tok.true.clean.en", 'w', encoding='utf-8') as f:
            f.writelines(lines["Europarl.en-fr.en"][:100_000] + lines["EMEA.en-fr.en"][:10_000])
        with open("../../TRAIN_data/Europarl_EMEA_train_100k_10k.tok.true.clean.fr", 'w', encoding='utf-8') as f:
            f.writelines(lines["Europarl.en-fr.fr"][:100_000] + lines["EMEA.en-fr.fr"][:10_000])

    print("Data preparation complete.")

    session = WSLSession()
    # Run initial commands
    session.run_command("conda activate env_opennmt")
    session.run_command("onmt_build_vocab -config run_1.yaml -n_sample 10000")
    session.run_command("onmt_train -config run_1.yaml")
    steps = 10000
    session.run_command(f"onmt_translate -model run_1/run/model_step_{steps}.pt -src TEST_data/Europarl_in_domain.tok.true.clean.fr -output run_1/pred_{steps}.txt -gpu 0 -verbose")
    session.run_command("./multi_bleu.pl ../data/Europarl_test_500.tok.true.clean.fr < ../data/pred_5000.txt")

    # Close the session when done
    session.close()