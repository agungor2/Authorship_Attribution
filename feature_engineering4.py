# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 22:04:29 2018

@author: mgungor
"""

import nltk 
import pandas as pd 
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
%matplotlib inline
from textblob import TextBlob
train_df = pd.read_csv("train1.csv")
test_df = pd.read_csv("test1.csv")
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
eng_stopwords = set(stopwords.words("english"))
## Number of unique words in the text ##
train_df["num_unique_words"] = train_df["text"].apply(lambda x: len(set(str(x).split())))
test_df["num_unique_words"] = test_df["text"].apply(lambda x: len(set(str(x).split())))
## Number of characters in the text ##
train_df["num_chars"] = train_df["text"].apply(lambda x: len(str(x)))
test_df["num_chars"] = test_df["text"].apply(lambda x: len(str(x)))
## Number of stopwords in the text ##
train_df["num_stopwords"] = train_df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test_df["num_stopwords"] = test_df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
##Number of Adjectives
train_df["num_adjs"] = train_df["text"].apply(lambda x: len([word for word, pos in nltk.pos_tag(tokenizer.tokenize(x)) if is_adj(pos) and len(word) > 1]))
test_df["num_adjs"] = test_df["text"].apply(lambda x: len([word for word, pos in nltk.pos_tag(tokenizer.tokenize(x)) if is_adj(pos) and len(word) > 1]))
##Number of Verbs
train_df["num_verbs"] = train_df["text"].apply(lambda x: len([word for word, pos in nltk.pos_tag(tokenizer.tokenize(x)) if is_verb(pos) and len(word) > 1]))
test_df["num_verbs"] = test_df["text"].apply(lambda x: len([word for word, pos in nltk.pos_tag(tokenizer.tokenize(x)) if is_verb(pos) and len(word) > 1]))
##Number of Nouns
train_df["num_nouns"] = train_df["text"].apply(lambda x: len([word for word, pos in nltk.pos_tag(tokenizer.tokenize(x)) if is_noun(pos) and len(word) > 1]))
test_df["num_nouns"] = test_df["text"].apply(lambda x: len([word for word, pos in nltk.pos_tag(tokenizer.tokenize(x)) if is_noun(pos) and len(word) > 1]))
##Polarity
train_df['polarity'] = train_df.text.apply(lambda row: TextBlob(row).sentiment[0])
test_df['polarity'] = train_df.text.apply(lambda row: TextBlob(row).sentiment[0])
##Subjectivity
train_df['subjectivity'] = train_df.text.apply(lambda row: TextBlob(row).sentiment[1])
test_df['subjectivity'] = train_df.text.apply(lambda row: TextBlob(row).sentiment[1])
import numpy as np
train_s = np.zeros((len(train_df.text), 3))
for i in range(len(train_df.text)):
    text_tmp = train_df["text"][i]
    tokens_tmp = tokenizer.tokenize(text_tmp)
    pos_list_tmp = nltk.pos_tag(tokens_tmp)
    #Check adjectives
    adj_tmp = len([word for word, pos in pos_list_tmp if is_adj(pos) and len(word) > 1])
    noun_tmp = len([word for word, pos in pos_list_tmp if is_noun(pos) and len(word) > 1])
    verb_tmp = len([word for word, pos in pos_list_tmp if is_verb(pos) and len(word) > 1])
    train_s[i,0] = int(adj_tmp)
    train_s[i,1] = int(noun_tmp)
    train_s[i,2] = int(verb_tmp)  
    print(i)
#Test Data
test_s = np.zeros((len(test_df.text), 3))
for i in range(len(test_df.text)):
    text_tmp = test_df["text"][i]
    tokens_tmp = tokenizer.tokenize(text_tmp)
    pos_list_tmp = nltk.pos_tag(tokens_tmp)
    #Check adjectives
    adj_tmp = len([word for word, pos in pos_list_tmp if is_adj(pos) and len(word) > 1])
    noun_tmp = len([word for word, pos in pos_list_tmp if is_noun(pos) and len(word) > 1])
    verb_tmp = len([word for word, pos in pos_list_tmp if is_verb(pos) and len(word) > 1])
    test_s[i,0] = int(adj_tmp)
    test_s[i,1] = int(noun_tmp)
    test_s[i,2] = int(verb_tmp)  
    print(i)