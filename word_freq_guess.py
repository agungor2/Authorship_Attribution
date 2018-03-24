# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 17:20:01 2018

@author: mgungor
"""

import nltk
import pandas as pd 
import numpy as np
import scipy.io as sc
from sklearn.metrics import f1_score

texts = pd.read_csv("train1.csv")

texts.head()

byAuthor = texts.groupby("author")

### Tokenize (split into individual words) our text

# word frequency by author
wordFreqByAuthor = nltk.probability.ConditionalFreqDist()

# for each author...
for name, group in byAuthor:
    # get all of the sentences they wrote and collapse them into a
    # single long string
    print(name)
    sentences = group['text'].str.cat(sep = ' ')
    
    # convert everything to lower case (so "The" and "the" get counted as 
    # the same word rather than two different words)
    sentences = sentences.lower()
    
    # split the text into individual tokens    
    tokens = nltk.tokenize.word_tokenize(sentences)
    
    # calculate the frequency of each token
    frequency = nltk.FreqDist(tokens)

    # add the frequencies for each author to our dictionary
    wordFreqByAuthor[name] = (frequency)
    
# see how often each author says "blood"
for i in wordFreqByAuthor.keys():
    print("blood: " + str(i))
    print(wordFreqByAuthor[i].freq('blood'))
    
# One way to guess authorship is to use the joint probabilty that each 
# author used each word in a given sentence.
#Test it on test data
test_df = pd.read_csv("test1.csv")
#Take 20 words and consider 100 batch of paragraphs
Batch_paragraphs = 100
all_prob = np.zeros((len(test_df.text), 50))
all_prob_prior = np.zeros((len(test_df.text), 50))
predicted_author_id = np.zeros((len(test_df.text), 1))
predicted_author_prior_id = np.zeros((len(test_df.text), 1))
#We can take the prior probablity of each author same 
#We can also take it as the frequecny text number of each author

prior_prob = {}
for i in texts.author:
    prior_prob[i] = len(texts[texts.author==i])/float(len(texts))

for k in range(len(test_df.text)):
    testSentence = test_df.text[k]
    # and then lowercase & tokenize our test sentence
    preProcessedTestSentence = nltk.tokenize.word_tokenize(testSentence.lower())
    # For each author...
    for i in wordFreqByAuthor.keys():
        # for each word in our test sentence...
        tmp_prob = 0
        for l in range(Batch_paragraphs):
            tmp = 1
            tmp2 = int(len(test_df.text[k].split())/Batch_paragraphs)
            for j  in range(tmp2):
                # find out how frequently the author used that word
                wordFreq = wordFreqByAuthor[i].freq(preProcessedTestSentence[l*tmp2+j])
                # and add a very small amount to every prob. so none of them are 0
                smoothedWordFreq = wordFreq + 0.000001
                tmp *= smoothedWordFreq
            tmp_prob += np.log(tmp)
        #save probablities authorwise
        all_prob[k][i-1] = tmp_prob
        #Consider Prior Prob as author Freq in the training set
        all_prob_prior[k][i-1] = tmp_prob*prior_prob[i]
    #Save the predicted author id
    predicted_author_id[k] = np.argmax(all_prob[k])+1
    #Save the predicted author id when considered prior probablity
    predicted_author_prior_id[k] = np.argmax(all_prob_prior[k])+1
    print(k)


#Now we can check the f1_score
test_author = sc.loadmat("test_author.mat")["test_author"]
print("F1 score for equal Prior Prob")
print(f1_score(test_author, predicted_author_id, average='micro'))
print("F1 score for Training Author Distribution as Prior Prob")
print(f1_score(test_author, predicted_author_prior_id, average='micro'))