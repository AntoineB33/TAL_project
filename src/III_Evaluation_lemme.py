import nltk
from nltk.stem import WordNetLemmatizer

from II_Evaluation_forme_flechie import *

if __name__ == "__main__":
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    print("love :", lemmatizer.lemmatize("result", wordnet.VERB))