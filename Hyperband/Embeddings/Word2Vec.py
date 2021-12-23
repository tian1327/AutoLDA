
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import os
import numpy as np
import nltk

# nltk.download('stopwords')
# nltk.download('punkt')


def getAllText(allfiles):
    res = []
    for file in allfiles:
        with open(file,"r") as f:
            listOfWord = f.read().split()
            res.append([word.lower() for word in listOfWord if word.isalpha()])
    return res            

def trainWord2Vec(allText):
    model = Word2Vec(sentences=allText, vector_size=100, window=5, min_count=1, workers=8)
    model.save("./Word2Vec_save/word2vec.model")
    print("Model saved.")
    # Store just the words + their trained embeddings.
    word_vectors = model.wv
    word_vectors.save("./Word2Vec_save/word2vec.wordvectors")
    print("Word vectors saved.")
    with open("./Word2Vec_save/word2vec.vocab", "w") as f:
        for word in word_vectors.key_to_index.keys():
            f.write(word + "\n")
    print("Vocab saved.")
    return model

def loadWord2Vec():
    cwd = os.getcwd()
    model = Word2Vec.load(cwd +"/Embeddings/Word2Vec_save/word2vec.model")
    word_vectors = model.wv
    return word_vectors

def getWord2Vec(word_vectors, word):
    try:
        return word_vectors[word]
    except KeyError:
        print("warning: key <{}> not found in W2V vectors, return all zeros.".format(word))
        return np.zeros(100)


# trainWord2Vec(allText)

# wv = loadWord2Vec()
# print(wv.vocab.keys())

# print(getWord2Vec(loadWord2Vec(), "covid"))

word_vectors = loadWord2Vec()

def genEmbeddings_W2V(keyword):
    
    emb = getWord2Vec(word_vectors, keyword)
    return emb

if __name__ == '__main__':
    allfiles = []
    cwd = os.getcwd()
    for root, dirs, files in os.walk(cwd + "/../Transcripts/", topdown=False):
        for name in files:
            if name.endswith(".txt") and "checkpoint" not in name:
                allfiles.append(os.path.join(root, name))

    allText = getAllText(allfiles)
    # print(len(allText))


