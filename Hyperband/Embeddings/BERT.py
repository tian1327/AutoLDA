#!/usr/bin/env python
# coding: utf-8


import os


import nltk
#nltk.download('stopwords')
#nltk.download('punkt')



# Ref: https://github.com/arushiprakash/MachineLearning/blob/main/BERT%20Word%20Embeddings.ipynb
# https://towardsdatascience.com/3-types-of-contextualized-word-embeddings-from-bert-using-transfer-learning-81fcefe3fe6d


from transformers.modeling_bert import BertModel, BertForMaskedLM

from transformers import BertTokenizer
# , BertModel
import pandas as pd
import numpy as np
import nltk
import torch



# Loading the pre-trained BERT model
###################################
# Embeddings will be derived from
# the outputs of this model
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True,
                                  )

# Setting up the tokenizer
###################################
# This is the same tokenizer that
# was used in the model to generate 
# embeddings to ensure consistency
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')






def bert_text_preparation(text, tokenizer):
    """Preparing the input for BERT
    
    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.
    
    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids
        
    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids
    
    
    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors
    
def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model
    
    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids
    
    Returns:
        list: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token
    
    """
    
    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]

    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings


def genEmbeddings_BERT(text):
    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
    list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
    # return list_token_embeddings
    # Find the position 'bank' in list of tokens
    
    if text not in tokenized_text:
        # print("BERT Embedding generation failed for ",text)
        # print("tokenized_text=",tokenized_text)
        return list_token_embeddings[1]
        # return []
    word_index = tokenized_text.index(text)
    # Get the embedding for bank
    word_embedding = list_token_embeddings[word_index]
    return word_embedding


if __name__ == "__main__":
    genEmbeddings_BERT("test")

# # texts = ["bank",
# #          "The river bank was flooded.",
# #          "The bank vault was robust.",
# #          "He had to bank on her for support.",
# #          "The bank was out of money.",
# #          "The bank teller was a man."]



# for text in texts:
#     tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
#     list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
    
#     # Find the position 'bank' in list of tokens
#     word_index = tokenized_text.index('bank')
#     # Get the embedding for bank
#     word_embedding = list_token_embeddings[word_index]

#     target_word_embeddings.append(word_embedding)


# from scipy.spatial.distance import cosine

# # Calculating the distance between the
# # embeddings of 'bank' in all the
# # given contexts of the word

# list_of_distances = []
# for text1, embed1 in zip(texts, target_word_embeddings):
#     for text2, embed2 in zip(texts, target_word_embeddings):
#         cos_dist = 1 - cosine(embed1, embed2)
#         list_of_distances.append([text1, text2, cos_dist])

# distances_df = pd.DataFrame(list_of_distances, columns=['text1', 'text2', 'distance'])


# # In[8]:


# distances_df[distances_df.text1 == 'bank']


# # In[12]:


# len(target_word_embeddings[0])


# # In[ ]:




