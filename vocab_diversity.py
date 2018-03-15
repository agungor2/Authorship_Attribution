# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 18:58:40 2018

@author: mgungor
"""
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
#3 Author dataset
train_df = pd.read_csv("train.csv")
train_df["num_unique_words"] = train_df["text"].apply(lambda x: len(set(str(x).split()))/len(str(x).split()))

#Overall vocabulary mean density for EAP, MWS, HPL
EAP = sum(train_df[train_df.author=="EAP"]['num_unique_words'])/len(train_df[train_df.author=="EAP"])
MWS = sum(train_df[train_df.author=="MWS"]['num_unique_words'])/len(train_df[train_df.author=="MWS"])
HPL = sum(train_df[train_df.author=="HPL"]['num_unique_words'])/len(train_df[train_df.author=="HPL"])


objects = ('EAP', 'MWS', 'HPL')
y_pos = np.arange(len(objects))
performance = [EAP, MWS, HPL]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Diversity Ratio')
plt.title('Author Word Richness')
 
plt.show()
###############################################################################
#50 authors dataset
train_df = pd.read_csv("train.csv",encoding = "ISO-8859-1")
train_df["num_unique_words"] = train_df["text"].apply(lambda x: len(set(str(x).split()))/len(str(x).split()))

objects = {}
for author_i in train_df.author.unique():
    objects[author_i] = sum(train_df[train_df.author==author_i]['num_unique_words'])/len(train_df[train_df.author==author_i])
    
plt.bar(range(len(objects)), list(objects.values()), align='center')
plt.xticks(range(len(objects)), list(objects.keys()))

plt.show()

print(max(objects, key=objects.get))
print(min(objects, key=objects.get))