# AutoLDA

Project for CSCE676 Data Mining course (Fall 2021). The github used during development can be found [here](https://github.com/jncsw/AutoLDA).

In this project, we implemented different hyperparameter searching methods to find the best hyperparameters for LDA, including **Hyperband, Grid Search, Random Search**. Specifically, we implemented Hyperband with LDA using number of iterations as resources. The results comparing against Random Search shows Hyperband achieves better score and clustering results.

Different implementation schemes of Hyperband with LDA were explored:

1. Using **data (# of documents)** as resources. This scheme allocates more training data to most promising configs. However, tests show that this does not work well because optimal LDA params should change with dataset (# of docs).

2. Using **# of iterations** as resouces, and perplexity as metric to quantitatively evaluate the goodness of LDA. This schemes splits data into training data and test data. The perplexity of test data is used as evaluation metric to filter the good configs. However, the perplexity is a biased score which is strongly affected by the number of topics chosen.

3. Using **# of iterations** as resources. This scheme uses full data to train the LDA with given iterations. Different embedding methods were used to calculate the embedding score. The locally-trained W2V outperformed the pretrained GLOVE, ELMO, BERT. It is believed that further fine-tuning these models will give better results than W2V.

Random Search and Grid Search were implemented as a baseline to compare with Hyperband. The results on our data shows that using the W2V score, Hyperband can find better hyperparameters than random search. It also yields great clustering results.

More details can be found in the [poster](https://sites.google.com/view/autolda/poster?authuser=0) and [website](https://sites.google.com/view/autolda/home).

## Hyperband + LDA
To run Hyperband with LDA using W2V embeddings:
```console
python main.py results_hb_W2V.pkl W2V
```

To show the best 10 configurations with its topic_words with given iterations:
```console
python show_results.py results_hb_W2V.pkl 10 W2V
```

To run the selected top1 config with full 81 resources to get the final LDA results:
```console
python run_configs.py 1 W2V
```

To plot the score vs. time of each embedding schemes:
```console
python plot.py
```

## Environment Setup for Embeddings
Load pretrained embedding models:

1. For GLOVE, download the pretrained model to folder `./Embeddings/GLOVE_pretrained/`, then run `GLOVE.py` to save the loaded model to pkl file

```console
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
```
```console
unzip glove.840B.300d.zip 
```
```console
python GLOVE.py 
```
