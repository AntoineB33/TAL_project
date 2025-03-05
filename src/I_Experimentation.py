import os
import subprocess
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

# Set Moses home directory (relative to the project location)
MOUSE_HOME = "/home/semmar/Training/SMT"
MOSES_HOME = "../../../../../Training/SMT" # TO_EDIT_BEFORE_DELIVERY


def getLines(lines, input_path):
    # process the content of the file if not already processed
    if not lines[input_path]:
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                lines[input_path] = f.readlines()
        except FileNotFoundError:
            print(f"File {input_path} not found.")
            sys.exit(1)

def tokenize(input_file, output_file, lang):
    tokenizer = Path(MOSES_HOME) / "mosesdecoder/scripts/tokenizer/tokenizer.perl"
    cmd = ["perl", str(tokenizer), "-l", lang]
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        subprocess.run(cmd, stdin=infile, stdout=outfile, check=True)

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

def clean_corpus(base_name, lang1, lang2, output_base, min_len, max_len):
    cleaner = Path(MOSES_HOME) / "mosesdecoder/scripts/training/clean-corpus-n.perl"
    cmd = ["perl", str(cleaner), base_name, lang1, lang2, output_base, 
           str(min_len), str(max_len)]
    subprocess.run(cmd, check=True)

def prepare_data(lines: dict[str: list[str]], inputPath: str, outputPath: str, outputName: str, start = 0, end = -1, tokenize = True, overwrite = False, is_train_100k_10k = False):
    if overwrite or not (Path(outputPath + outputName + ".en").exists() and Path(outputPath + outputName + ".fr").exists()) or is_train_100k_10k and not (Path(outputPath + "Europarl_EMEA_train_100k_10k.tok.true.clean.en").exists() and Path(outputPath + "Europarl_EMEA_train_100k_10k.tok.true.clean.fr").exists()):
        for lang in ["en", "fr"]:
            # a. Splitting data
            print("Splitting data...")
            getLines(lines, "Europarl.en-fr."+lang)
            with open(outputName+"."+lang, 'w', encoding='utf-8') as f:
                if end == -1:
                    f.writelines(lines[inputPath+"."+lang][start:])
                else:
                    f.writelines(lines[inputPath+"."+lang][start:end])

            if tokenize:
                # b. Tokenization
                print("Tokenizing...")
                tokenize(outputName+"."+lang, outputName+".tok."+lang, lang)

            # c. Truecasing
            print("Truecasing...")
            # c.1 Training truecase models
            train_truecaser(outputName+".tok."+lang, "truecase-model_train."+lang)

            # c.2 Applying truecasing
            apply_truecaser("truecase-model_train."+lang, outputName+".tok."+lang, 
                            outputName+".tok.true."+lang)
            
        # d. Cleaning corpus
        print("Cleaning corpus...")
        clean_corpus(outputName+".tok.true", "fr", "en", 
                    outputPath + outputName + ".tok.true.clean", 1, 80)
        
        if is_train_100k_10k:
            for lang in ["en", "fr"]:
                if is_train_100k_10k:
                    # e. Combining corpora
                    print("Combining corpora...")
                    train_lines = []
                    with open("../../TRAIN_data/Europarl_train_100k.tok.true.clean."+lang, 'r', encoding='utf-8') as f:
                        train_lines += f.readlines()
                    with open("../../TRAIN_data/EMEA_train_10k.tok.true.clean."+lang, 'r', encoding='utf-8') as f:
                        train_lines += f.readlines()
                    with open("../../TRAIN_data/Europarl_EMEA_train_100k_10k.tok.true.clean."+lang, 'w', encoding='utf-8') as f:
                        f.writelines(train_lines)


if __name__ == "__main__":
    args = sys.argv[1:]

    lines = {}
    
    # Change to data directory
    os.chdir(Path("data/I_Experimentation/data_preparation/TRAIN_DEV_TEST_joints"))
    
    prepare_data(lines, "Europarl_train_10k", "", "Europarl_train_10k", tokenize=False)
    prepare_data(lines, "Europarl_dev_1k", "", "Europarl_dev_1k", tokenize=False)
    prepare_data(lines, "Europarl_test_500", "", "Europarl_test_500", tokenize=False)

    # Change back to the original directory
    os.chdir(Path("../../../../.."))