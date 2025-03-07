# TAL_project


Pour préparer les données d'entraînement, d'évaluation et de test :
- placer le dossier Europarl.en-fr.txt, contenant Europarl.en-fr.fr, dans data/
- faire la même chose avec EMEA.en-fr.txt
- se connecter à l'environement de développement : conda activate env_opennmt
- Installer les librairies Python : pip install --file requirements.txt
- Changer MOSES_HOME dans prepare_data.py si besoin
- lancer par exemple : python src/I_Experimentation.py



Pour entraîner les modèles sur les données préparées :

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

cd ../I_Experimentation

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












to debug in VSCode :
- go to wsl
- activate the environment with : conda activate env_opennmt
- run : which python
- copy the given path directory and paste it in the "python" attribute of your launch.json file