# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 23:29:26 2018

@author: Mecit
"""
import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords

eng_stopwords = set(stopwords.words("english"))

train_df = pd.read_csv("train.csv")
## Number of punctuations in the text ##
train_df["num_punctuations"] =train_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
## Number of title case words in the text ##
train_df["num_words_upper"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
train_df["num_words_title"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the text ##
train_df["mean_word_len"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

## Number of stopwords in the text ##
train_df["num_stopwords"] = train_df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

df = pd.DataFrame(columns = ['EAP','HPL','MWS'])
#Number of punctuations density
df.index.names = ['num_punctuations']
df.loc['num_punctuations'] = [sum(train_df[train_df.author==x]["num_punctuations"])/len(train_df[train_df.author==x]) for x in ['EAP','HPL','MWS']]
#Number of title case density
df.index.names = ['num_words_upper']
df.loc['num_words_upper'] = [sum(train_df[train_df.author==x]["num_words_upper"])/len(train_df[train_df.author==x]) for x in ['EAP','HPL','MWS']]
#Number of title case density
df.index.names = ['num_words_title']
df.loc['num_words_title'] = [sum(train_df[train_df.author==x]["num_words_title"])/len(train_df[train_df.author==x]) for x in ['EAP','HPL','MWS']]
#Average length of the words density
df.index.names = ['mean_word_len']
df.loc['mean_word_len'] = [sum(train_df[train_df.author==x]["mean_word_len"])/len(train_df[train_df.author==x]) for x in ['EAP','HPL','MWS']]
#Number of stopwords density
df.index.names = ['num_stopwords']
df.loc['num_stopwords'] = [sum(train_df[train_df.author==x]["num_stopwords"])/len(train_df[train_df.author==x]) for x in ['EAP','HPL','MWS']]
            