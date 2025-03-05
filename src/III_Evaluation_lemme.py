from II_Evaluation_forme_flechie import *

if __name__ == "__main__":
    args = sys.argv[1:]

    # Ensure the necessary NLTK data is downloaded
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    prepare_all_data("III_Evaluation_lemme", *args)