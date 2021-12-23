#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 00:35:52 2021

@author: tian
"""

import matplotlib.pyplot as plt
import pandas as pd

def plot_score(data, lab):
   
   x=data['Time(s)']
   y=data['score']
   
   fig, ax = plt.subplots(figsize=(6,6))
   
   ax.plot(x, y, label=lab, marker='o', color='red')
   
   plt.legend()
   plt.xlim([0,1400])
   plt.ylim([0.2,0.6])
   
   ax.set(xlabel='Time (s)', ylabel='Embedding_Score', title='')
   # plt.xticks(np.arange(0, 50, step=5))
   # ax.grid()
   
   # fig.savefig("test.png")
   plt.show()


data = pd.read_csv('score_vs_time_W2V.csv')
plot_score(data, 'HB_W2V')

data = pd.read_csv('score_vs_time_GLOVE.csv')
plot_score(data, 'HB_GLOVE')

data = pd.read_csv('score_vs_time_BERT.csv')
plot_score(data, 'HB_BERT')

data = pd.read_csv('score_vs_time_ELMO.csv')
plot_score(data, 'HB_ELMO')