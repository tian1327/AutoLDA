"function (and parameter space) definitions for hyperband"
"LDA"

from common_defs import *
from hyperopt.pyll.stochastic import sample
# from load_data_lda import train_data
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.base import BaseEstimator, ClassifierMixin
from random import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from lda_metric import embedding_distance
import pickle

# define search space
space = {
         'max_df': hp.uniform('maxdf', 0.6, 0.8),
         'min_df': hp.uniform('mindf', 0.02, 0.2),
         # 'max_df': hp.choice('maxdf', (0.6, 0.6)),
         # 'min_df': hp.choice('mindf', (0.05, 0.05)),
         
         # 'topic_number': 7
         'topic_number': 5 + hp.randint('tn',10),
         
         'doc_topic_prior': hp.uniform('alpha', 0.01, 2),
         'topic_word_prior': hp.uniform('beta', 0.01, 2),
         
         # 'learning_method': hp.choice('lm', ('online', 'batch')),
         # 'learning_decay': hp.uniform('kappa', 0.51, 1.0),
         # 'learning_offset': 1 + hp.randint('tau_0', 20),
         # 'batch_size': hp.choice( 'bs', ( 16, 32, 64, 128, 256 )),
         # 'max_iter': hp.choice('max_iter',(100,200))
         # 'max_iter': hp.choice('max_iter',(5, 10, 20))
    }

print('loading data')
with open('data.pkl','rb') as pk:
    data = pickle.load(pk)


# print('loading train_data')
# with open('train_data.pkl','rb') as pk:
#     train_data = pickle.load(pk)

# print('loading test_data')
# with open('test_data.pkl','rb') as pk:
#     test_data = pickle.load(pk)


def get_params():
    params = sample(space)
    return handle_integers(params)

def print_params(params):
    pprint({ k : v for k, v in params.items()})
    return None

def try_params(n_iterations, params, emb_model):

    print_params(params)
    
    # run LDA on data
    lda = LDA_classifier(n_iterations, params)
    
    t_w, n_iter, d_t = lda.fit(data)    
    
    lda_score = lda.semantic_score(t_w, emb_model)
    
    # perplexity_train = lda.score(train_data) 
    # perplexity_test = lda.score(test_data) # calculate the perplexity on the held-out test data
    
    # print('perplexity_train (lda.score) = {}'.format(perplexity_train))
    # print('perplexity_test = {:.4f}'.format(perplexity_test))
    # print('n_iter = {}'.format(n_iter))
    
    print('lda_score =', lda_score)
    
    return {'lda_score': lda_score,  
            'topic_keywords': t_w, 
            'n_iter':n_iter, 
            'doc_topic_distr': d_t}


# def show_topics(lda, n_words=10):
#     vectorizer, lda_model = best_LDA.get_model()
    
#     keywords = np.array(vectorizer.get_feature_names())
    
#     topic_keywords = []
#     
#     for topic_weights in lda_model.components_:
#         top_keyword_locs = (-topic_weights).argsort()[:n_words]
#         topic_keywords.append(keywords.take(top_keyword_locs))
#     
#     for i in range(0, len(topic_keywords)):
#         print("Topic " + str(i))
#         print(list(topic_keywords[i]))
#     
#     return topic_keywords

class LDA_classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iterations, params):
        self.max_df = params['max_df']
        self.min_df = params['min_df']
        self.topic_number = params['topic_number']
        self.doc_topic_prior = params['doc_topic_prior']
        self.topic_word_prior = params['topic_word_prior']
    
        # self.learning_decay = params['learning_decay']
        # self.learning_offset = params['learning_offset']
        # self.batch_size = params['batch_size']

        self.max_iter = n_iterations        
        
    def fit(self, train_data):
        # print('fitting:', self.max_df, self.min_df, self.topic_number)
        self.vectorizer = CountVectorizer(max_df=self.max_df, min_df=self.min_df)
        
        self.lda_model = LatentDirichletAllocation(n_components=self.topic_number,
                                                   doc_topic_prior=self.doc_topic_prior,
                                                   topic_word_prior=self.topic_word_prior,
                                                   learning_method='online', 
                                                   # learning_decay = self.learning_decay,
                                                   # learning_offset = self.learning_offset,
                                                   # batch_size = self.batch_size,                                                 
                                                    max_iter=self.max_iter,
                                                   random_state=100)
        
        data_vectorized = self.vectorizer.fit_transform(train_data)
        self.lda_model.fit(data_vectorized)
        
        n_iter = self.lda_model.n_iter_
        # perplexity_train = self.lda_model.bound_
        
        print('n_iter =', n_iter)
        # print("perplexity_train = {:.4f}".format(perplexity_train))

        # params = self.lda_model.get_params()
        # print("===== params:\n", params)

        # perplexity = self.lda_model.perplexity(data_vectorized)
        # print("===== perplexity:",perplexity)
        
        # return the doc_topic_distr
        doc_topic_distr = self.lda_model.transform(data_vectorized)
        # print(doc_topic_distr.shape)
        # print(doc_topic_distr[0,:])

        # return the topic_keywords
        keywords = np.array(self.vectorizer.get_feature_names())
        topic_keywords = []
        n_words = 10
        
        for topic_weights in self.lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        
        # for i in range(0, len(topic_keywords)):
        #     print("Topic " + str(i))
        #     print(list(topic_keywords[i]))
        
        return topic_keywords, n_iter, doc_topic_distr

    def predict(self, texts):
        text_vectorized = self.vectorizer.transform(texts)
        return self.lda_model.transform(text_vectorized)

    def score(self, train_data):
        # score = self.lda_model.perplexity(self.data_vectorized)
        tmp = self.vectorizer.transform(train_data)
        score = self.lda_model.perplexity(tmp)
        
        # print('perplexity score =', score)
        return score
    
    
    def semantic_score(self, topic_keywords, emb_model):
        
        score = 0
        
        score = embedding_distance(topic_keywords, emb_model)
        # score = random()
        
        return score

    def get_model(self):
        # print('the final model:', self.max_df, self.min_df, self.topic_number)
        return (self.vectorizer, self.lda_model)