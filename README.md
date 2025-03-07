# TAL_project


google doc : https://docs.google.com/document/d/1chrQMpC4Pnln_k_ZKWvhCj1Lqpx-1u8CuYIMA3qcRpE/edit?tab=t.0

to run the Python programs :
- place the Europarl.en-fr.txt folder, containing Europarl.en-fr.fr, in data/
- do the same with EMEA.en-fr.txt



- Se connecter à l'environement de développement :
conda activate env_opennmt

- Installer les librairies Python :
pip install --file requirements.txt


================ Exercice 3 ================================

cd data/III_Evaluation_lemme/

créer les vocabulaires :
onmt_build_vocab -config run_1_en_to_fr.yaml -n_sample 10000
onmt_build_vocab -config run_1_fr_to_en.yaml -n_sample 10000
onmt_build_vocab -config run_2_en_to_fr.yaml -n_sample 10000
onmt_build_vocab -config run_2_fr_to_en.yaml -n_sample 10000

entraîner les modèles :
onmt_train -config run_1_en_to_fr.yaml
onmt_train -config run_1_fr_to_en.yaml
onmt_train -config run_2_en_to_fr.yaml
onmt_train -config run_2_fr_to_en.yaml

traduire dans le domaine :
onmt_translate -model run_1_en_to_fr/run/model_step_1000.pt -src TEST_data/Europarl_test_in_500.tok.true.clean.en -output run_1_en_to_fr/pred_in_1000.txt -gpu 0 -verbose
onmt_translate -model run_1_fr_to_en/run/model_step_1000.pt -src TEST_data/Europarl_test_in_500.tok.true.clean.fr -output run_1_fr_to_en/pred_in_1000.txt -gpu 0 -verbose
onmt_translate -model run_1_en_to_fr/run/model_step_1000.pt -src TEST_data/Europarl_test_out_500.tok.true.clean.en -output run_1_en_to_fr/pred_out_1000.txt -gpu 0 -verbose
onmt_translate -model run_1_fr_to_en/run/model_step_1000.pt -src TEST_data/Europarl_test_out_500.tok.true.clean.fr -output run_1_fr_to_en/pred_out_1000.txt -gpu 0 -verbose
onmt_translate -model run_2_en_to_fr/run/model_step_1000.pt -src TEST_data/Europarl_test_in_500.tok.true.clean.en -output run_2_en_to_fr/pred_in_1000.txt -gpu 0 -verbose
onmt_translate -model run_2_fr_to_en/run/model_step_1000.pt -src TEST_data/Europarl_test_in_500.tok.true.clean.fr -output run_2_fr_to_en/pred_in_1000.txt -gpu 0 -verbose
onmt_translate -model run_2_en_to_fr/run/model_step_1000.pt -src TEST_data/Europarl_test_out_500.tok.true.clean.en -output run_2_en_to_fr/pred_out_1000.txt -gpu 0 -verbose
onmt_translate -model run_2_fr_to_en/run/model_step_1000.pt -src TEST_data/Europarl_test_out_500.tok.true.clean.fr -output run_2_fr_to_en/pred_out_1000.txt -gpu 0 -verbose

mesurer les tests dans le domaine :
../../src/multi_bleu.pl TEST_data/Europarl_test_in_500.tok.true.clean.fr < run_1_en_to_fr/pred_in_1000.txt
../../src/multi_bleu.pl TEST_data/Europarl_test_in_500.tok.true.clean.en < run_1_fr_to_en/pred_in_1000.txt
../../src/multi_bleu.pl TEST_data/Europarl_test_out_500.tok.true.clean.fr < run_1_en_to_fr/pred_out_1000.txt
../../src/multi_bleu.pl TEST_data/Europarl_test_out_500.tok.true.clean.en < run_1_fr_to_en/pred_out_1000.txt
../../src/multi_bleu.pl TEST_data/Europarl_test_in_500.tok.true.clean.fr < run_2_en_to_fr/pred_in_1000.txt
../../src/multi_bleu.pl TEST_data/Europarl_test_in_500.tok.true.clean.en < run_2_fr_to_en/pred_in_1000.txt
../../src/multi_bleu.pl TEST_data/Europarl_test_out_500.tok.true.clean.fr < run_2_en_to_fr/pred_out_1000.txt
../../src/multi_bleu.pl TEST_data/Europarl_test_out_500.tok.true.clean.en < run_2_fr_to_en/pred_out_1000.txt




================ Exercice 2 ================================

cd ../II_Evaluation_forme_flechie

créer les vocabulaires :
onmt_build_vocab -config run_1_en_to_fr.yaml -n_sample 10000
onmt_build_vocab -config run_1_fr_to_en.yaml -n_sample 10000
onmt_build_vocab -config run_2_en_to_fr.yaml -n_sample 10000
onmt_build_vocab -config run_2_fr_to_en.yaml -n_sample 10000

entraîner les modèles :
onmt_train -config run_1_en_to_fr.yaml
onmt_train -config run_1_fr_to_en.yaml
onmt_train -config run_2_en_to_fr.yaml
onmt_train -config run_2_fr_to_en.yaml

traduire dans le domaine :
onmt_translate -model run_1_en_to_fr/run/model_step_10000.pt -src TEST_data/Europarl_test_in_500.tok.true.clean.en -output run_1_en_to_fr/pred_in_10000.txt -gpu 0 -verbose
onmt_translate -model run_1_en_to_fr/run/model_step_10000.pt -src TEST_data/Europarl_test_out_500.tok.true.clean.en -output run_1_en_to_fr/pred_out_10000.txt -gpu 0 -verbose
onmt_translate -model run_1_fr_to_en/run/model_step_10000.pt -src TEST_data/Europarl_test_in_500.tok.true.clean.fr -output run_1_fr_to_en/pred_in_10000.txt -gpu 0 -verbose
onmt_translate -model run_1_fr_to_en/run/model_step_10000.pt -src TEST_data/Europarl_test_out_500.tok.true.clean.fr -output run_1_fr_to_en/pred_out_10000.txt -gpu 0 -verbose
onmt_translate -model run_2_en_to_fr/run/model_step_10000.pt -src TEST_data/Europarl_test_in_500.tok.true.clean.en -output run_2_en_to_fr/pred_in_10000.txt -gpu 0 -verbose
onmt_translate -model run_2_en_to_fr/run/model_step_10000.pt -src TEST_data/Europarl_test_out_500.tok.true.clean.en -output run_2_en_to_fr/pred_out_10000.txt -gpu 0 -verbose
onmt_translate -model run_2_fr_to_en/run/model_step_10000.pt -src TEST_data/Europarl_test_in_500.tok.true.clean.fr -output run_2_fr_to_en/pred_in_10000.txt -gpu 0 -verbose
onmt_translate -model run_2_fr_to_en/run/model_step_10000.pt -src TEST_data/Europarl_test_out_500.tok.true.clean.fr -output run_2_fr_to_en/pred_out_10000.txt -gpu 0 -verbose

mesurer les tests dans le domaine :
../../src/multi_bleu.pl TEST_data/Europarl_test_in_500.tok.true.clean.fr < run_1_en_to_fr/pred_in_10000.txt
../../src/multi_bleu.pl TEST_data/Europarl_test_out_500.tok.true.clean.fr < run_1_en_to_fr/pred_out_10000.txt
../../src/multi_bleu.pl TEST_data/Europarl_test_in_500.tok.true.clean.en < run_1_fr_to_en/pred_in_10000.txt
../../src/multi_bleu.pl TEST_data/Europarl_test_out_500.tok.true.clean.en < run_1_fr_to_en/pred_out_10000.txt
../../src/multi_bleu.pl TEST_data/Europarl_test_in_500.tok.true.clean.fr < run_2_en_to_fr/pred_in_10000.txt
../../src/multi_bleu.pl TEST_data/Europarl_test_out_500.tok.true.clean.fr < run_2_en_to_fr/pred_out_10000.txt
../../src/multi_bleu.pl TEST_data/Europarl_test_in_500.tok.true.clean.en < run_2_fr_to_en/pred_in_10000.txt
../../src/multi_bleu.pl TEST_data/Europarl_test_out_500.tok.true.clean.en < run_2_fr_to_en/pred_out_10000.txt



================ Exercice 1 ================================

cd "/mnt/c/Users/abarb/Documents/travail/Polytech Paris Saclay/cours/et5/TAL/projet/TAL_project"
conda activate env_opennmt
cd data/I_Experimentation

créer les vocabulaires :
onmt_build_vocab -config TRAIN_DEV_TEST_joints_en_to_fr.yaml -n_sample 10000
onmt_build_vocab -config TRAIN_DEV_TEST_joints_fr_to_en.yaml -n_sample 10000

entraîner les modèles :
onmt_train -config TRAIN_DEV_TEST_joints_en_to_fr.yaml
onmt_train -config TRAIN_DEV_TEST_joints_fr_to_en.yaml

traduire dans le domaine :
onmt_translate -model TRAIN_DEV_TEST_joints_en_to_fr/run/model_step_2500.pt -src TEST_data/Europarl_test_500.tok.true.clean.en -output TRAIN_DEV_TEST_joints_en_to_fr/pred_2500.txt -gpu 0 -verbose
onmt_translate -model TRAIN_DEV_TEST_joints_fr_to_en/run/model_step_2500.pt -src TEST_data/Europarl_test_500.tok.true.clean.fr -output TRAIN_DEV_TEST_joints_fr_to_en/pred_2500.txt -gpu 0 -verbose

mesurer les tests dans le domaine :
../../src/multi_bleu.pl TEST_data/Europarl_test_500.tok.true.clean.fr < TRAIN_DEV_TEST_joints_en_to_fr/pred_2500.txt
../../src/multi_bleu.pl TEST_data/Europarl_test_500.tok.true.clean.en < TRAIN_DEV_TEST_joints_fr_to_en/pred_2500.txt













onmt_translate -model TRAIN_DEV_TEST_joints_en_to_fr/run/model_en_to_fr_step_4000.pt -src TEST_data/Europarl_test_500.tok.true.clean.fr -output TRAIN_DEV_TEST_joints_en_to_fr/pred_in_4000.txt -gpu 0 -verbose
../../src/multi_bleu.pl TEST_data/Europarl_test_500.tok.true.clean.en < TRAIN_DEV_TEST_joints_en_to_fr/pred_in_4000.txt



to debug in VSCode :
- go to wsl
- activate the environment with : conda activate env_opennmt
- run : which python
- copy the given path directory and paste it in the "python" attribute of your launch.json file