#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 19:52:23 2021

@author: tian
"""

import matplotlib.pyplot as plt
import pandas as pd

def plot_score(data1, data2,lab):
   
   x=data1['Time(s)']
   y=data1['score']
   
   x2=data2['Time(s)']
   y2=data2['score']
   
   fig, ax = plt.subplots(figsize=(6,6))
   
   ax.plot(x, y, label=lab[0], marker='o', color='red')
   ax.plot(x2, y2, label=lab[1], marker='o', color='blue')
   
   plt.legend()
   plt.xlim([0,1400])
   plt.ylim([0.2,0.6])
   
   ax.set(xlabel='Time (s)', ylabel='Embedding Score', title='')
   # plt.xticks(np.arange(0, 50, step=5))
   # ax.grid()
   
   # fig.savefig("test.png")
   plt.show()


data1 = pd.read_csv('score_vs_time_W2V.csv')
data2 = pd.read_csv('random_w2v.csv')
plot_score(data1, data2, ['HB_W2V','RS_W2V'])

data1 = pd.read_csv('score_vs_time_GLOVE.csv')
data2 = pd.read_csv('random_glove.csv')
plot_score(data1,data2, ['HB_GLOVE','RS_GLOVE'])

data1 = pd.read_csv('score_vs_time_BERT.csv')
data2 = pd.read_csv('random_bert.csv')
plot_score(data1, data2, ['HB_BERT','RS_BERT'])

data1 = pd.read_csv('score_vs_time_ELMO.csv')
data2 = pd.read_csv('random_elmo.csv')
plot_score(data1, data2, ['HB_ELMO','RS_ELMO'])