from sklearn.model_selection import train_test_split

from I_Experimentation import *


def prepare_all_data(data_folder, *args):
    overwrite = "overwrite" in args

    is_exo_3 = data_folder == "III_Evaluation_lemme"
    lines = {}
    
    # Change to data directory
    os.chdir(Path(f"data/{data_folder}/data_preparation"))
    
    prepare_data(lines, "../TRAIN_data/", "Europarl_train_100k", length = 100_000, is_exo_3 = is_exo_3)

    prepare_data(lines, "../DEV_data/", "Europarl_dev_3750", start = 100_000, length = 3_750, is_exo_3 = is_exo_3)

    if "no_new_test" not in args or not (Path("../TEST_data/Europarl_test_in_500.tok.true.clean.en").exists() and Path("../TEST_data/Europarl_test_in_500.tok.true.clean.en").exists()):
        # for the test, getting random pairs of lines inside the domain using train_test_split
        print("preparing test data for inside the domain")
        getLines(lines, "../../Europarl.en-fr.txt/Europarl.en-fr.en")
        getLines(lines, "../../Europarl.en-fr.txt/Europarl.en-fr.fr")
        _, test_en_lines, _, test_fr_lines = train_test_split(lines["../../Europarl.en-fr.txt/Europarl.en-fr.en"][103_750:], lines["../../Europarl.en-fr.txt/Europarl.en-fr.fr"][103_750:], test_size=500)
        with open("Europarl_test_in_500.en", 'w', encoding='utf-8') as f:
            f.writelines(test_en_lines)
        with open("Europarl_test_in_500.fr", 'w', encoding='utf-8') as f:
            f.writelines(test_fr_lines)
        prepare_data(lines, "../TEST_data/", "Europarl_test_in_500", overwrite = True, is_exo_3 = is_exo_3, inputPath="Europarl_test_in_500")

        # for the test, getting random pairs of lines outside the domain using train_test_split
        print("preparing test data for outside the domain")
        getLines(lines, "../../EMEA.en-fr.txt/EMEA.en-fr.en")
        getLines(lines, "../../EMEA.en-fr.txt/EMEA.en-fr.fr")
        _, test_en_lines, _, test_fr_lines = train_test_split(lines["../../EMEA.en-fr.txt/EMEA.en-fr.en"][10_000:], lines["../../EMEA.en-fr.txt/EMEA.en-fr.fr"][10_000:], test_size=500)
        with open("Europarl_test_out_500.en", 'w', encoding='utf-8') as f:
            f.writelines(test_en_lines)
        with open("Europarl_test_out_500.fr", 'w', encoding='utf-8') as f:
            f.writelines(test_fr_lines)
        prepare_data(lines, "../TEST_data/", "Europarl_test_out_500", overwrite = True, is_exo_3 = is_exo_3, inputPath="Europarl_test_out_500")

    prepare_data(lines, "../TRAIN_data/", "EMEA_train_10k", start = 13_750, length = 10_000, is_train_100k_10k = True, is_exo_3 = is_exo_3, inputPath="../../EMEA.en-fr.txt/EMEA.en-fr")
    

    print("Data preparation complete.")

if __name__ == "__main__":
    args = sys.argv[1:]
    args = ["overwrite"]
    #args = ["no_new_test"]
    #args = ["test_not_random"]

    prepare_all_data("II_Evaluation_forme_flechie", *args)





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