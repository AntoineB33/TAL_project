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

def prepare_data(lines: dict[str: list[str]], inputPath: str, outputName: str, outputPath, start = 0, end = -1, overwrite = False):
    if overwrite or not (Path(outputPath + outputName + ".en").exists() and Path(outputPath + outputName + ".fr").exists()):
        for lang in ["en", "fr"]:
            # a. Splitting data
            print("Splitting data...")
            getLines(lines, "Europarl.en-fr."+lang)
            with open(outputName+"."+lang, 'w', encoding='utf-8') as f:
                if end == -1:
                    f.writelines(lines[inputPath+"."+lang][start:])
                else:
                    f.writelines(lines[inputPath+"."+lang][start:end])

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


if __name__ == "__main__":
    args = sys.argv[1:]
    #args = ["no_new_test"]
    #args = ["test_not_random"]

    lines = {}
    
    # Change to data directory
    os.chdir(Path("data/II_Evaluation_forme_flechie/data_preparation"))
    
    prepare_data(lines, "Europarl.en-fr.txt/Europarl.en-fr.txt", "Europarl_train_100k", "../../TRAIN_data/", end = 100_000)

    prepare_data(lines, "Europarl.en-fr.txt/Europarl.en-fr.txt", "Europarl_dev_3750", "../../DEV_data/", start = 100_000, end = 103_750)

    if "no_new_test" not in args or not (Path("../../TEST_data/Europarl_test_in_500.tok.true.clean.en").exists() and Path("../../TEST_data/Europarl_test_in_500.tok.true.clean.en").exists()):
        # for the test, getting random pairs of lines inside the domain using train_test_split
        print("preparing test data for inside the domain")
        getLines(lines, "Europarl.en-fr.en")
        getLines(lines, "Europarl.en-fr.fr")
        _, test_en_lines, _, test_fr_lines = train_test_split(lines["Europarl.en-fr.en"][:103_750], lines["Europarl.en-fr.fr"][:103_750], test_size=500)
        with open("../../TEST_data/Europarl_test_in_500.tok.true.clean.en", 'w', encoding='utf-8') as f:
            f.writelines(test_en_lines)
        getLines(lines, "EMEA.en-fr.en")
        getLines(lines, "EMEA.en-fr.fr")
        with open("../../TEST_data/Europarl_test_in_500.tok.true.clean.fr", 'w', encoding='utf-8') as f:
            f.writelines(test_fr_lines)

        # for the test, getting random pairs of lines outside the domain using train_test_split
        print("preparing test data for outside the domain")
        _, test_en_lines, _, test_fr_lines = train_test_split(lines["Europarl.en-fr.en"][103_750:], lines["Europarl.en-fr.fr"][103_750:], test_size=500)
        with open("../../TEST_data/Europarl_test_out_500.tok.true.clean.en", 'w', encoding='utf-8') as f:
            f.writelines(test_en_lines)
        with open("../../TEST_data/Europarl_test_out_500.tok.true.clean.fr", 'w', encoding='utf-8') as f:
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


    # commandes wsl pour utiliser entraîner et évaluer les modèles

    #conda activate env_opennmt
    #cd data/II_Evaluation_forme_flechie
    #onmt_translate -model run_1/run/model_step_2500.pt -src TEST_data/Europarl_test_in_500.tok.true.clean.fr -output run_1/pred_in_2500.txt -gpu 0 -verbose
    #../../src/multi_bleu.pl TEST_data/Europarl_test_in_500.tok.true.clean.en < run_1/pred_in_2500.txt



    # tentative de faire ça automatiquement (ne marche pas)

    # # Run 1
    # stdout, stderr, return_code = run_wsl_command("onmt_build_vocab -config run_1.yaml -n_sample 10000")
    # stdout, stderr, return_code = run_wsl_command("onmt_train -config run_1.yaml")
    # steps = 10000
    # stdout, stderr, return_code = run_wsl_command(f"onmt_translate -model run_1/run/model_step_{steps}.pt -src TEST_data/Europarl_test_in_500.tok.true.clean.fr -output run_1/pred_in_{steps}.txt -gpu 0 -verbose")
    # stdout, stderr, return_code = run_wsl_command(f"./multi_bleu.pl TEST_data/Europarl_test_in_500.tok.true.clean.fr < run_1/pred_in_{steps}.txt")
    # stdout, stderr, return_code = run_wsl_command(f"onmt_translate -model run_1/run/model_step_{steps}.pt -src TEST_data/Europarl_test_out_500.tok.true.clean.fr -output run_1/pred_out_{steps}.txt -gpu 0 -verbose")
    # stdout, stderr, return_code = run_wsl_command(f"./multi_bleu.pl TEST_data/Europarl_test_out_500.tok.true.clean.fr < run_1/pred_out_{steps}.txt")

    # # Run 2
    # stdout, stderr, return_code = run_wsl_command("onmt_build_vocab -config run_2.yaml -n_sample 10000")
    # stdout, stderr, return_code = run_wsl_command("onmt_train -config run_2.yaml")
    # stdout, stderr, return_code = run_wsl_command(f"onmt_translate -model run_2/run/model_step_{steps}.pt -src TEST_data/Europarl_test_in_500.tok.true.clean.fr -output run_2/pred_in_{steps}.txt -gpu 0 -verbose")
    # stdout, stderr, return_code = run_wsl_command(f"./multi_bleu.pl TEST_data/Europarl_test_in_500.tok.true.clean.fr < run_2/pred_in_{steps}.txt")
    # stdout, stderr, return_code = run_wsl_command(f"onmt_translate -model run_2/run/model_step_{steps}.pt -src TEST_data/Europarl_test_out_500.tok.true.clean.fr -output run_2/pred_out_{steps}.txt -gpu 0 -verbose")
    # stdout, stderr, return_code = run_wsl_command(f"./multi_bleu.pl TEST_data/Europarl_test_out_500.tok.true.clean.fr < run_2/pred_out_{steps}.txt")