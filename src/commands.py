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

# Utility function to split files
def split_files(input_path, train_lines, dev_lines, train_output, dev_output):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Write training split
    with open(train_output, 'w', encoding='utf-8') as f:
        f.writelines(lines[:train_lines])
    
    # Write dev split
    with open(dev_output, 'w', encoding='utf-8') as f:
        f.writelines(lines[train_lines:train_lines+dev_lines])

def write_pairs(pairs, en_path, fr_path):
    """Write paired sentences to respective files."""
    en = [p[0] for p in pairs]
    fr = [p[1] for p in pairs]
    with open(en_path, 'w', encoding='utf-8') as f_en, open(fr_path, 'w', encoding='utf-8') as f_fr:
        f_en.writelines(en)
        f_fr.writelines(fr)

def split_paired_data(en_path, fr_path, splits, output_files, random_state=42):
    """Split and shuffle paired data using train_test_split."""
    with open(en_path, 'r', encoding='utf-8') as f_en, open(fr_path, 'r', encoding='utf-8') as f_fr:
        en = f_en.readlines()
        fr = f_fr.readlines()
    
    assert len(en) == len(fr), "Mismatched lines in input files"
    paired = list(zip(en, fr))
    
    # Split into train, dev, test_in
    train, remaining = train_test_split(paired, train_size=splits[0], random_state=random_state)
    dev, test_in = train_test_split(remaining, train_size=splits[1], test_size=splits[2], random_state=random_state)
    
    # Write splits to files
    write_pairs(train, output_files[0][0], output_files[0][1])
    write_pairs(dev, output_files[1][0], output_files[1][1])
    write_pairs(test_in, output_files[2][0], output_files[2][1])

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
    # args = ["--in"]
    test_in = "--in" in args
    test_out = "--out" in args
    if not test_in and not test_out:
        test_out = True
        test_in = True

    # check if the data has already been processed
    if not Path("../../TRAIN_data/Europarl_train_100k.tok.true.clean.en").exists() or not Path("../../TRAIN_data/Europarl_train_100k.tok.true.clean.fr").exists():

        # a. Split files
        print("Splitting files...")
        split_files("Europarl.en-fr.en", 10000, 3750, "Europarl_train_100k.en", "Europarl_dev_3750.en")
        split_files("Europarl.en-fr.fr", 10000, 3750, "Europarl_train_100k.fr", "Europarl_dev_3750.fr")

        # b. Tokenization
        print("Tokenizing...")
        tokenize("Europarl_train_100k.en", "Europarl_train_100k.tok.en", "en")
        tokenize("Europarl_train_100k.fr", "Europarl_train_100k.tok.fr", "fr")
        tokenize("Europarl_dev_3750.en", "Europarl_dev_3750.tok.en", "en")
        tokenize("Europarl_dev_3750.fr", "Europarl_dev_3750.tok.fr", "fr")

        # c. Truecasing
        print("Truecasing...")
        # c.1 Training truecase models
        # Note: The original commands contain a typo here (3450 vs 3750)
        train_truecaser("Europarl_train_100k.tok.en", "truecase-model_train.en")
        train_truecaser("Europarl_train_100k.tok.fr", "truecase-model_train.fr")
        train_truecaser("Europarl_dev_3750.tok.en", "truecase-model_dev.en")  # Fixed filename
        train_truecaser("Europarl_dev_3750.tok.fr", "truecase-model_dev.fr")  # Fixed filename

        # c.2 Applying truecasing
        apply_truecaser("truecase-model_train.en", "Europarl_train_100k.tok.en", 
                        "Europarl_train_100k.tok.true.en")
        apply_truecaser("truecase-model_train.fr", "Europarl_train_100k.tok.fr", 
                        "Europarl_train_100k.tok.true.fr")
        apply_truecaser("truecase-model_dev.en", "Europarl_dev_3750.tok.en", 
                        "Europarl_dev_3750.tok.true.en")  # Fixed filename
        apply_truecaser("truecase-model_dev.fr", "Europarl_dev_3750.tok.fr", 
                        "Europarl_dev_3750.tok.true.fr")  # Fixed filename

        print("Cleaning corpus...")
        clean_corpus("Europarl_train_100k.tok.true", "fr", "en", 
                    "../../TRAIN_data/Europarl_train_100k.tok.true.clean", 1, 80)
        clean_corpus("Europarl_dev_3750.tok.true", "fr", "en", 
                    "../../DEV_data/Europarl_dev_3750.tok.true.clean", 1, 80)

        print("Processing complete!")

    if "--not_rand_test" not in args or test_in and not (Path("../../TEST_data/Europarl_test_in_500.en").exists() and Path("../../TEST_data/Europarl_test_in_500.fr")) or test_out and not (Path("../../TEST_data/Europarl_test_out_500.en").exists() and Path("../../TEST_data/Europarl_test_out_500.fr")):
        # Split test data
        split_random("Europarl.en-fr.en", 13750, 500, "Europarl_test_in_500.en", "Europarl_test_out_500.en")
        split_random("Europarl.en-fr.fr", 13750, 500, "Europarl_test_in_500.fr", "Europarl_test_out_500.fr")
