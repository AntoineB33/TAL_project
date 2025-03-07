import sys
import subprocess
from pathlib import Path

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pathlib import Path

from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer


# Set Moses home directory (relative to the project location)
MOSES_HOME = "/home/semmar/Training/SMT"


def getLines(lines, input_path):
    # process the content of the file if not already processed
    if input_path not in lines:
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                lines[input_path] = f.readlines()
        except FileNotFoundError:
            print(f"File {input_path} not found.")
            sys.exit(1)

def tokenize(input_file, output_file, lang):
    """Tokenize the input text using the Moses tokenizer."""
    tokenizer = Path(MOSES_HOME) / "mosesdecoder/scripts/tokenizer/tokenizer.perl"
    cmd = ["perl", str(tokenizer), "-l", lang]
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        # result = subprocess.run("dir", shell=True, check=True, text=True, capture_output=True)
        # print(f"Command: ls")
        # print(f"Output: {result.stdout}")
        subprocess.run(cmd, stdin=infile, stdout=outfile, check=True)

def train_truecaser(corpus_path, model_path):
    """Train a truecaser model using the provided corpus."""
    trainer = Path(MOSES_HOME) / "mosesdecoder/scripts/recaser/train-truecaser.perl"
    cmd = ["perl", str(trainer), "--model", model_path, "--corpus", corpus_path]
    subprocess.run(cmd, check=True)

def apply_truecaser(model_path, input_path, output_path):
    """Apply truecasing using the trained truecaser model."""
    truecaser = Path(MOSES_HOME) / "mosesdecoder/scripts/recaser/truecase.perl"
    cmd = ["perl", str(truecaser), "--model", model_path]
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        subprocess.run(cmd, stdin=infile, stdout=outfile, check=True)

def clean_corpus(base_name, lang1, lang2, output_base, min_len, max_len):
    """Clean parallel corpora by removing excessively long or short sentences."""
    cleaner = Path(MOSES_HOME) / "mosesdecoder/scripts/training/clean-corpus-n.perl"
    cmd = ["perl", str(cleaner), base_name, lang1, lang2, output_base, 
           str(min_len), str(max_len)]
    subprocess.run(cmd, check=True)

def download_nltk_resources():
    """Download necessary NLTK resources for lemmatization."""
    global wordnet_lemmatizer, french_lemmatizer
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    
    nltk.download('verbnet')
    
    wordnet_lemmatizer = WordNetLemmatizer()
    french_lemmatizer = FrenchLefffLemmatizer()

def lemmatize_text(lang: str, text: str) -> str:
    """Lemmatize the input text based on the specified language."""
    global wordnet_lemmatizer, french_lemmatizer
    lemmatizer = None
    if lang == "en":
        lemmatizer = wordnet_lemmatizer
    elif lang == "fr":
        lemmatizer = french_lemmatizer
    else:
        raise ValueError(f"Language '{lang}' not supported")
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def prepare_data(lines: dict[str: list[str]], outputPath: str, outputName: str, start = 0, length = -1, overwrite = False, is_train_100k_10k = False, inputPath = "../../Europarl.en-fr.txt/Europarl.en-fr", is_exo_3 = False):
    """Prepare training data by performing tokenization, truecasing, and cleaning."""
    if overwrite or not (Path(outputPath + outputName + ".tok.true.clean.en").exists() and Path(outputPath + outputName + ".tok.true.clean.fr").exists()) or is_train_100k_10k and not (Path(outputPath + "Europarl_EMEA_train_100k_10k.tok.true.clean.en").exists() and Path(outputPath + "Europarl_EMEA_train_100k_10k.tok.true.clean.fr").exists()):
        for lang in ["en", "fr"]:
            if not (start == 0 and length == -1):
                # a. Splitting data
                print("Splitting data...")
                getLines(lines, inputPath+"."+lang)
                with open(outputName+"."+lang, 'w', encoding='utf-8') as f:
                    if length == -1:
                        f.writelines(lines[inputPath+"."+lang][start:])
                    else:
                        f.writelines(lines[inputPath+"."+lang][start:start + length])

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
            
            if is_exo_3:
                # d. Lemmatization (only for English)
                print("Lemmatizing...")
                with open(outputName + ".tok.true." + lang, 'r', encoding='utf-8') as f:
                    lemmatized_lines = [lemmatize_text(lang, line) for line in f.readlines()]
                with open(outputName + ".tok.true.lem." + lang, 'w', encoding='utf-8') as f:
                    f.writelines(lemmatized_lines)
            
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
                    with open("../TRAIN_data/Europarl_train_100k.tok.true.clean."+lang, 'r', encoding='utf-8') as f:
                        train_lines += f.readlines()
                    with open("../TRAIN_data/EMEA_train_10k.tok.true.clean."+lang, 'r', encoding='utf-8') as f:
                        train_lines += f.readlines()
                    with open("../TRAIN_data/Europarl_EMEA_train_100k_10k.tok.true.clean."+lang, 'w', encoding='utf-8') as f:
                        f.writelines(train_lines)