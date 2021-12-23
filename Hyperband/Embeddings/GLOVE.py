import os
import numpy as np
import pickle
# import tqdm

def load_GLOVE():
    model = 'glove_pretrained_840b_300d.pkl'
    print("loading GLOVE pretrained model ......")
    with open('./Embeddings/GLOVE_pretrained/'+model,'rb') as pk:
        glove_emb = pickle.load(pk)
    print('GLOVE loaded.\n')

    return glove_emb

def genEmbeddings_GLOVE(keyword):
    # print('gen GLOVE')
    word_embedding = [0 for i in range(300)]
    if keyword in glove_emb:
        word_embedding = glove_emb[keyword]
    else:
        print('--'*10, keyword, 'not found in GLOVE!')

    return word_embedding

glove_emb = load_GLOVE()

if __name__ == "__main__":
    path_to_glove_file = "./GLOVE_pretrained/glove.840B.300d.txt"
    embeddings_dict = {}
    with open(path_to_glove_file) as f:
        for line in f:
            value = line.split(' ')
            word = value[0]
            coefs = np.array(value[1:], dtype = 'float32')
            embeddings_dict[word] = coefs

    print('save GLOVE embeddings_dict to pkl ......')
    with open('./GLOVE_pretrained/glove_pretrained_840b_300d.pkl','wb') as f:
        pickle.dump(embeddings_dict, f)

