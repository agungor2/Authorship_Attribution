# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 17:22:14 2018

@author: mgungor
"""
import pandas as pd 
from collections import Counter
 import matplotlib.pyplot as plt
train_df = pd.read_csv("train.csv")
from nltk.tokenize.moses import MosesTokenizer, MosesDetokenizer
t, d = MosesTokenizer(), MosesDetokenizer()

author_wise = {}
for author_i in train_df.author.unique():
    print(author_i)
    at = train_df[train_df.author == author_i].text.values
    count_not = 0
    count = 0
    count_s =0
    count_is = 0
    for i in range(len(at)):
        print(i)
        if(i!=3808 and i!=4141):
            tokens = t.tokenize(at[i])
            count_s += Counter(tokens)["&apos;s"]
            count_is += Counter(tokens)["is"]
            count_not += Counter(tokens)["not"]
            count += Counter(tokens)["&apos;t"]
    author_wise[author_i] = [count_s, count_is, count_not, count]
    
d={'apos-s': {'EAP':384, 'HPL':612,  'MWS':352},
 'is': {'EAP':1639, 'HPL':364, 'MWS':681},
 'not': {'EAP':1252,'HPL':834, 'MWS':1105},
 'apos-not':{'EAP':87,'HPL':186,'MWS':0}}

pd.DataFrame(d).plot(kind='bar')
plt.show()