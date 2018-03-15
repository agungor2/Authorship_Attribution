# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:40:45 2018

@author: mgungor
"""

from nltk.util import ngrams
from collections import Counter
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
df = pd.read_csv("train.csv", index_col="id")
test = pd.read_csv("test.csv", index_col="id")
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

###############################################################################
def Ngram1(text, n):
    n_grams = ngrams((text), n)
    return [ ' '.join(grams) for grams in n_grams]

def NgramFreq(text,n,num):
    result = Ngram1(text,n)
    result_count = Counter(result)
    df = pd.DataFrame.from_dict(result_count, orient='index')
    df = df.rename(columns={'index':'words', 0:'frequency'}) 
    return df.sort_values(["frequency"],ascending=[0])[:num]

def preprocessing(data):
    txt = data.str.lower().str.cat(sep=' ') #1
    words = tokenizer.tokenize(txt) #2
    words = [w for w in words if not w in stop_words] #3
    return words

def GramTable(x, gram, length):
    out = pd.DataFrame(index=None)
    for i in gram:
        table = pd.DataFrame(NgramFreq(preprocessing(df[df.author == x]['text']),i,length).reset_index())
        table.columns = ["{}-Gram".format(i),"Occurence"]
        out = pd.concat([out, table], axis=1)
    return out
###############################################################################
#Test it on the three author dataset
print(GramTable(x="EAP", gram=[1,2,3,4], length=5))
print(GramTable(x="MWS", gram=[1,2,3,4], length=5))
print(GramTable(x="HPL", gram=[1,2,3,4], length=5))


###############################################################################
#Train it on the 50 authors dataset
df = pd.read_csv("train.csv", encoding = "ISO-8859-1")
test = pd.read_csv("test.csv",encoding = "ISO-8859-1")
print(GramTable(x=1, gram=[1,2,3,4], length=5))
print(GramTable(x=2, gram=[1,2,3,4], length=5))
print(GramTable(x=6, gram=[1,2,3,4], length=5))
