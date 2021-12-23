from Embeddings.BERT import genEmbeddings_BERT
from Embeddings.ELMo import genEmbeddings_ELMo
from Embeddings.Word2Vec import genEmbeddings_W2V
from Embeddings.GLOVE import genEmbeddings_GLOVE
# from tqdm import tqdm
import numpy as np


# Topic 0
# ['baby', 'car', 'vlog', 'camera', 'house', 'family', 'ready', 'girl', 'tell', 'channel']
# Topic 1
# ['walk', 'pretty', 'store', 'open', 'close', 'place', 'nice', 'empty', 'park', 'show']
# Topic 2
# ['apartment', 'move', 'stuff', 'room', 'love', 'new', 'show', 'top', 'big', 'cute']
# Topic 3
# ['home', 'stay', 'life', 'week', 'watch', 'mean', 'live', 'read', 'keep', 'talk']
# Topic 4
# ['man', 'school', 'happen', 'talk', 'home', 'mean', 'week', 'money', 'basically', 'store']
# Topic 5
# ['mask', 'wear', 'hand', 'wash', 'egg', 'mean', 'case', 'stuff', 'hopefully', 'happen']
# Topic 6
# ['use', 'love', 'show', 'watch', 'morning', 'eat', 'week', 'do', 'hair', 'stuff']
# Topic 7
# ['stuff', 'eat', 'picture', 'mom', 'man', 'play', 'room', 'bad', 'wait', 'wake']
# Topic 8
# ['call', 'give', 'case', 'check', 'point', 'phone', 'eat', 'far', 'week', 'hear']
# Topic 9
# ['walk', 'happen', 'line', 'hair', 'show', 'pretty', 'use', 'talk', 'keep', 'stuff']

InputData = [
['baby', 'car', 'vlog', 'camera', 'house', 'family', 'ready', 'girl', 'tell', 'channel'],
['walk', 'pretty', 'store', 'open', 'close', 'place', 'nice', 'empty', 'park', 'show'],
['apartment', 'move', 'stuff', 'room', 'love', 'new', 'show', 'top', 'big', 'cute'],
['home', 'stay', 'life', 'week', 'watch', 'mean', 'live', 'read', 'keep', 'talk'],
['man', 'school', 'happen', 'talk', 'home', 'mean', 'week', 'money', 'basically', 'store'],
['mask', 'wear', 'hand', 'wash', 'egg', 'mean', 'case', 'stuff', 'hopefully', 'happen'],
['use', 'love', 'show', 'watch', 'morning', 'eat', 'week', 'do', 'hair', 'stuff'],
['stuff', 'eat', 'picture', 'mom', 'man', 'play', 'room', 'bad', 'wait', 'wake'],
['call', 'give', 'case', 'check', 'point', 'phone', 'eat', 'far', 'week', 'hear'],
['walk', 'happen', 'line', 'hair', 'show', 'pretty', 'use', 'talk', 'keep', 'stuff'] ]



def GenEmb(InputData, model):

    # print("++++++++++ Model = ", model)
    # print("++++++++++ InputData = ", InputData)
    if model == "ELMO":
        return genEmbeddings_ELMo(InputData)

    AllEmb = []
    
    # for topic in tqdm(InputData):
    for topic in (InputData):
        topicEmb = []
        for keyword in topic:
            if model == "BERT":
                emb = genEmbeddings_BERT(keyword)
            elif model == "GLOVE":
                emb = genEmbeddings_GLOVE(keyword)
            elif model == "W2V":                
                emb = genEmbeddings_W2V(keyword)
            # elif model == "ELMO":
            #     emb = genEmbeddings_ELMo(keyword)
            topicEmb.append(emb)
        AllEmb.append(topicEmb)

    return AllEmb

def dist(a,b):
    return np.linalg.norm(np.asarray(a)-np.asarray(b))
    

def Calc_Dist(AllEmb):
    flatterned = []
    for topic in AllEmb:
        flatterned.append([item for sublist in topic for item in sublist])
    res = -1
    for x in range(len(flatterned)):
        for y in range(len(flatterned)):
            if x==y:
                continue
            cur_dist = dist(flatterned[x],flatterned[y])
            res = max(res,cur_dist)
    return res
                    
            
    


if __name__ == "__main__":
    # print(genEmbeddings_BERT(("test"))
    print("Generating BERT Embeddings...")
    AllEmb = GenEmb_BERT(InputData)
    print("Calculating distance...")
    print("Score = ", Calc_Dist(AllEmb))
    
    print("All Done.")
            
            
    