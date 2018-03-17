# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 10:14:45 2018

@author: Mecit
"""

import nltk 
import pandas as pd 
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv("train.csv")
#Check if it's adjective 
def is_adj(pos):
    result = False
    if pos in ('JJ','JJR','JJS'):
        result = True
    return result
#Check if it's verb
def is_verb(pos):
    result = False
    if pos in ('VB','VBD','VBG','VBN','VBP','VBZ'):
        result = True
    return result
#Check if it's noun 
def is_noun(pos):
    result = False
    if pos in ('NN','NNP','NNPS','NNS'):
        result = True
    return result
#Preprocess the data for authowise
tokenizer = RegexpTokenizer(r'\w+')
def Syntactic_features(author):
    print(author)
    df_tmp = df[df['author'] == author]
    text_tmp = df_tmp['text'].str.cat(sep = ' ').lower()
    tokens_tmp = tokenizer.tokenize(text_tmp)
    pos_list_tmp = nltk.pos_tag(tokens_tmp)
    #Check adjectives
    adj_tmp = [word for word, pos in pos_list_tmp if is_adj(pos) and len(word) > 1]
    #Freq distribution for adjectives
    freq_tmp = nltk.FreqDist(adj_tmp)
    print("Most Common 10 adjectives")
    print(freq_tmp.most_common(10))
    #number of adjectives divided by total number of words
    adj_tmp2 = len(adj_tmp)/len(tokens_tmp)
    #Check for verbs
    verb_tmp = [word for word, pos in pos_list_tmp if is_verb(pos) and len(word) > 1] 
    #Frequency of the adjectives
    freq_tmp_verb = nltk.FreqDist(verb_tmp)
    print("Most Common 10 verbs")
    print(freq_tmp_verb.most_common(10))
    verb_tmp2 = len(verb_tmp)/len(tokens_tmp)
    #Check for nouns
    noun_tmp = [word for word, pos in pos_list_tmp if is_noun(pos) and len(word) > 1]
    freq_tmp_noun = nltk.FreqDist(noun_tmp) 
    print("Most Common 10 nouns")
    print(freq_tmp_noun.most_common(10))
    noun_tmp2 = len(noun_tmp)/len(tokens_tmp)
    return([adj_tmp2,verb_tmp2, noun_tmp2])
###############################################################################
#Apply it on the 3 authors dataset
overall_result = {}
for author_i in df.author.unique():
    tmp = Syntactic_features(author_i)
    overall_result[author_i] = tmp

syntactic_features = ["adjective", "verb", "noun"]
d = {}
for element in range(len(syntactic_features)):
    d_tmp = {}
    for author_i in df.author.unique():
        d_tmp[author_i] = overall_result[author_i][element]
    d[syntactic_features[element]] = d_tmp 

pd.DataFrame(d).plot(kind='bar')
plt.show()

###############################################################################
#Apply it on the 50 authors dataset
df = pd.read_csv("train.csv",encoding = "ISO-8859-1")
author_list = [1,2,3,4,6]
overall_result = {}
for author_i in author_list:
    tmp = Syntactic_features(author_i)
    overall_result[author_i] = tmp

syntactic_features = ["adjective", "verb", "noun"]
d = {}
for element in range(len(syntactic_features)):
    d_tmp = {}
    for author_i in author_list:
        d_tmp[author_i] = overall_result[author_i][element]
    d[syntactic_features[element]] = d_tmp 

pd.DataFrame(d).plot(kind='bar')
plt.show()