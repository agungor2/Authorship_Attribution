# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 01:52:08 2018

@author: mgungor
"""
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))
import time
import multiprocessing
from gensim.models import Word2Vec
from sklearn.metrics import f1_score, accuracy_score
import scipy.io as sc
cores = multiprocessing.cpu_count()
import pandas as pd
import numpy as np
train = pd.read_csv('train1.csv')
test = pd.read_csv('test1.csv')
ytrain = train.author.values
test_author = sc.loadmat("test_author.mat")["test_author"]
#Unique authors in the training set
unique_author = np.unique(train.author.values)    
w=500
s=250
all_probs = np.zeros((len(unique_author), len(test.text)))
for k in range(len(unique_author)):
    author = unique_author[k]
    sentences = train[train.author == author]["text"].ravel().tolist()
    new_sentences = []
    for x in sentences:
        new_sentences.append(x.split())
        
    ## create a w2v learner 
    size = 300
    basemodel = Word2Vec(
        workers=multiprocessing.cpu_count(), size=size, # use your cores
        iter=3, # iter = sweeps of SGD through the data; more is better
        hs=1, negative=0 # we only have scoring for the hierarchical softmax setup
        )
    basemodel.build_vocab(new_sentences)
    print(basemodel)
    basemodel.train(new_sentences, total_examples=len(new_sentences),epochs=basemodel.epochs)
    print(author)
    tic = time.clock()
    for i in range(len(test.text)):
        tmp_txt = test.text[i]
        tmp_txt = tmp_txt.split()
        b = len(tmp_txt)
        tmp2 = int(b/s-1)
        if tmp2 ==0:
            tmp_txt3 = ''
            for x in tmp_txt:
                if x not in eng_stopwords:
                   tmp_txt3 += x  
            llhd = basemodel.score([tmp_txt3])[0]
            if np.isnan(np.log(llhd)):
                all_probs[k][i] =0
            else:
                all_probs[k][i] =np.log(llhd)
        else:
            probs = np.zeros((1,int(b/s-1)))
            for j in range(int(b/s-1)):
                tmp_txt2 = tmp_txt[s*j:s*j+w]
                tmp_txt3 = ''
                for x in tmp_txt2:
                    if x not in eng_stopwords:
                       tmp_txt3 += x  
                probs[0,j] = basemodel.score([tmp_txt3])[0]
            #Standardize it
            lhd = np.exp(probs[0,:] - probs[0,:].max(axis=0))
            #Log likelihood
            llhd = basemodel.score([tmp_txt3])[0]
            if np.isnan(np.log(llhd)):
                all_probs[k][i] =0
            else:
                llhd = np.sum(np.log(lhd[lhd!=0]))
                all_probs[k][i] =llhd
    toc = time.clock()
    print(toc - tic)
    
predicted_author = np.zeros((len(test.text),1))
#Now Loop through the whole sequence
#Label the max value as the prediction
for k in range(np.shape(all_probs)[1]):
    at = all_probs[:,k]/np.sum(all_probs[:,k])
    #Divide By the sum of all probabilities
    predicted_author[k] = unique_author[np.argmax(at)]
print("f1 Score")
print(f1_score(test_author, predicted_author, average='micro'))
print("Accuracy")
print(accuracy_score(test_author, predicted_author))    
##############################################################################
    #Let's change w and s
w=1000
s=500
all_probs = np.zeros((len(unique_author), len(test.text)))
for k in range(len(unique_author)):
    author = unique_author[k]
    sentences = train[train.author == author]["text"].ravel().tolist()
    new_sentences = []
    for x in sentences:
        new_sentences.append(x.split())
        
    ## create a w2v learner 
    size = 300
    basemodel = Word2Vec(
        workers=multiprocessing.cpu_count(), size=size, # use your cores
        iter=3, # iter = sweeps of SGD through the data; more is better
        hs=1, negative=0 # we only have scoring for the hierarchical softmax setup
        )
    basemodel.build_vocab(new_sentences)
    print(basemodel)
    basemodel.train(new_sentences, total_examples=len(new_sentences),epochs=basemodel.iter)
    print(author)
    tic = time.clock()
    for i in range(len(test.text)):
        tmp_txt = test.text[i]
        tmp_txt = tmp_txt.split()
        tmp_txt3 = ''
        for x in tmp_txt:
            if x not in eng_stopwords:
               tmp_txt3 += x  
        llhd = basemodel.score([tmp_txt3])[0]
        all_probs[k][i] =llhd
    toc = time.clock()
    print(toc - tic)
    
predicted_author = np.zeros((len(test.text),1))
#Now Loop through the whole sequence
#Label the max value as the prediction
for k in range(np.shape(all_probs)[1]):
    at = all_probs[:,k]/np.sum(all_probs[:,k])
    #Divide By the sum of all probabilities
    predicted_author[k] = unique_author[np.argmax(at)]
    
print("f1 Score")
print(f1_score(test_author, predicted_author, average='micro'))
print("Accuracy")
print(accuracy_score(test_author, predicted_author))