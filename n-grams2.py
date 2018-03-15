# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 18:18:16 2018

@author: mgungor
"""

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import nltk
from nltk.util import ngrams
%matplotlib inline

train = pd.read_csv("train.csv", encoding = "ISO-8859-1")
def generate_ngrams(text, n=2):
    words = text.split()
    iterations = len(words) - n + 1
    for i in range(iterations):
       yield words[i:i + n]
       
       
# DataFrame for Arthur Conan Doyle
ngrams = {}
for title in train[train.author==1]['text']:
        for ngram in generate_ngrams(title, 2):
            ngram = ' '.join(ngram)
            if ngram in ngrams:
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1

ngrams_acd_df = pd.DataFrame.from_dict(ngrams, orient='index')
ngrams_acd_df.columns = ['count']
ngrams_acd_df['author'] = 'Arthur Conen Doyle'
ngrams_acd_df.reset_index(level=0, inplace=True)

# DataFrame for William Carleton
ngrams = {}
for title in train[train.author==22]['text']:
        for ngram in generate_ngrams(title, 2):
            ngram = ' '.join(ngram)
            if ngram in ngrams:
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1

ngrams_wc_df = pd.DataFrame.from_dict(ngrams, orient='index')
ngrams_wc_df.columns = ['count']
ngrams_wc_df['author'] = 'William Carleton'
ngrams_wc_df.reset_index(level=0, inplace=True)

# DataFrame for Anne Manning
ngrams = {}
for title in train[train.author==24]['text']:
        for ngram in generate_ngrams(title, 2):
            ngram = ' '.join(ngram)
            if ngram in ngrams:
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1

ngrams_am_df = pd.DataFrame.from_dict(ngrams, orient='index')
ngrams_am_df.columns = ['count']
ngrams_am_df['author'] = 'Anne Manning'
ngrams_am_df.reset_index(level=0, inplace=True)       

print(ngrams_acd_df.sort_values(by='count', ascending=False).head(3))

print(ngrams_wc_df.sort_values(by='count', ascending=False).head(3))

print(ngrams_am_df.sort_values(by='count', ascending=False).head(3))


bigram_df = pd.concat([ngrams_acd_df.sort_values(by='count', ascending=False).head(20),
                        ngrams_wc_df.sort_values(by='count', ascending=False).head(20),
                        ngrams_am_df.sort_values(by='count', ascending=False).head(20)])
    
g = nx.from_pandas_dataframe(bigram_df,source='author',target='index')
plt.figure(figsize=(20, 20))
cmap = plt.cm.viridis_r
colors = [n for n in range(len(g.nodes()))]
plt.figure(figsize=(20, 20))
k = 0.35
pos=nx.spring_layout(g, k=k)
nx.draw_networkx(g,pos, node_size=bigram_df['count'].values*6, cmap = cmap, 
                 node_color=colors, edge_color='grey', font_size=15, width=3)
plt.show()
plt.axis("off")