# TAL_project


google doc : https://docs.google.com/document/d/1chrQMpC4Pnln_k_ZKWvhCj1Lqpx-1u8CuYIMA3qcRpE/edit?tab=t.0

to run the Python programs :
- place the Europarl.en-fr.txt folder, containing Europarl.en-fr.fr, at data/
- do the same with EMEA.en-fr.txt



conda activate env_opennmt

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


cd ../II_Evaluation_forme_flechie

créer les vocabulaires :
onmt_train -config run_1_en_to_fr.yaml
onmt_train -config run_1_fr_to_en.yaml
onmt_train -config run_2_en_to_fr.yaml
onmt_train -config run_2_fr_to_en.yaml




to debug in VSCode :
- go to wsl
- activate the environment with : conda activate env_opennmt
- run : which python
- copy the given path directory and paste it in the "python" attribute of your launch.json file