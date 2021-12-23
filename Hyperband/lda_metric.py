import numpy as np
# from gensim.models import LdaModel
from GenerateEmbeddings import GenEmb, Calc_Dist


def embedding_distance(topic_words, model):

    # allModel = ["BERT", "GLOVE", "W2V", "ELMo"]
    # for model in allModel:
    #     embeddings = GenEmb(topic_words, model)
    #     print("------ Embedding for ", model, "------", len(embeddings))
    #     # print("Calculating distance...")
    #     # print("Score = ", Calc_Dist(AllEmb))
    #     #score = Calc_Dist(AllEmb)

    embeddings = GenEmb(topic_words, model)
    # print(np.array(embeddings).shape)
    # distance
    score = cal_distance(embeddings, 'coh-dif')

    return score


def cal_distance(embeddings, method):
    """
    embeddings is a list of list of list
                    topic   words   embs
    """
    emb = np.array(embeddings)
    # print(embeddings)
    # print(embeddings.type)
    # print(emb.shape) # 10, 10, 300 for GLOVE

    num_topic = emb.shape[0]
    num_words = emb.shape[1]
    num_dim = emb.shape[2]

    score = 0

    if method == "coh-dif":

        # cal the coherence within the same topic
        topic_centers = np.mean(emb, axis=1) # average along the words

        # print(topic_centers)
        # print(topic_centers.shape) 

        coh_sum = 0
        for t in range(num_topic): # loop through each topic

            coh = 0
            coh_ct = 0
            # zero_ct = 0
            for w in range(num_words): # loop through each words in a topic
                word = emb[t,w,:]
                # if np.amax(word)==0:
                #     zero_ct += 1

                topic_center = topic_centers[t,:]
                coh += np.linalg.norm(word-topic_center)
                coh_ct += 1

            coh = coh / coh_ct
            coh_sum += coh
            # print('zero_ct =', zero_ct)

        print('coh_sum =',coh_sum)

        # cal the dif between each topic center and the global center
        global_center = np.mean(topic_centers, axis=0) # average along the topics
        # print("global_center =", global_center)
        # print("shape =", global_center.shape)

        dif_sum = 0
        for t in range(num_topic): # loop through each topic
            dif = np.linalg.norm(topic_centers[t,:]-global_center)
            dif_sum += dif

        print('dif_sum =', dif_sum)

        # center distance
        center_distance =  np.linalg.norm(topic_centers[0,:]-topic_centers[1,:])
        print('center_distance bt topic0 and topic1 =', center_distance)

        # cal the score dif_sum/coh_sum
        score = dif_sum / coh_sum

    return score


def jaccard(self,list1,list2):
    intersection=len(list(set(list1).intersection(list2)))
    union=(len(list1)+len(list2))-intersection
    return float(intersection)/union

def get_stability(self,topic_matrix):
    mean=[]
    row=topic_matrix.shape[0]
    for i in range(row):
        mean.append(np.mean(topic_matrix[i]))
    sum=0
    for i in range(row):
        sim=jaccard(row[i],mean)
        sum+=sim
    stability=sum/np.sum(topic_matrix)
    
    return stability


def coefficient_of_variation(self,list):
    mean=np.mean(list)
    std=np.std(list,ddof=0)
    cv=std/mean
    return cv

def get_variability(self,topic_matrix):
    row=topic_matrix.shape[0]
    cv_list=[]
    for i in range(row):
        cv=coefficient_of_variation(topic_matrix[i])
        cv_list.append(cv)
    variability=np.std(cv_list,ddof=0)
    return variability


def log_perplexity(self, chunk, total_docs=None):
    if total_docs is None:
        total_docs=len(chunk)
    corpus_words=sum(cnt for document in chunk for _,cnt in document)
    subsample_ratio=1.0*total_docs/len(chunk)
    perwordbound=self.bound(chunk,subsample_ratio=subsample_ratio)/(subsample_ratio * corpus_words)
    perplexity=np.exp2(-perwordbound)
    
    return perplexity

#calulate the perplexity of LDA model
# lda=LdaModel(common_corpus,num_topics=num_topic,id2word=idc,alpha='auto',chunksize=len(texts_all),iterations=20000)
# perplexity=lda.log_perplexity(common_corpus)


if __name__ == "__main__":
    # test = [[[0, 1, 2], [3, 4, 5]], 
    #         [[0, 1, 2], [3, 4, 5]],  
    #         [[0, 1, 2], [3, 4, 5]], 
    #         [[0, 1, 2], [3, 4, 5]]]
    # score = cal_distance(test, 'coh-dif')
    # print('score =', score)

    topic_words = [['us','irq','lead','work','govt','troop','turn','announce','bushfire','reach'],
                   ['baby', 'car', 'vlog', 'camera', 'house', 'family', 'ready', 'girl', 'tell', 'channel']
                ]

    topic_words = [['game','fish','moose','wildlife','hunting','bears','polar','bear','subsistence','management'],
                    ['gas','oil','pipeline','agia','project','natural','north','producers','companies','tax'] ]

    score = embedding_distance(topic_words,'GLOVE')
    print('score =', score)
