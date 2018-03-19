# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 21:01:00 2018

@author: mgungor
"""

from gensim.models import Word2Vec
import gensim
import multiprocessing
import pandas as pd
import numpy as np
df = pd.read_csv("train.csv", encoding = "ISO-8859-1")

def word_cos_distance(authorid, word):
    #Preperation of the text data for word2vec training
    sentences = df[df.author == authorid]["text"].ravel().tolist()
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
    print(basemodel)
    basemodel.build_vocab(new_sentences)
    basemodel.train(new_sentences, total_examples=len(new_sentences),epochs=basemodel.iter)
    return(basemodel[word])
    
def sentence_cos_distance(authorid, sentence):
    #Preperation of the text data for word2vec training
    sentences = df[df.author == authorid]["text"].ravel().tolist()
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
    print(basemodel)
    basemodel.build_vocab(new_sentences)
    basemodel.train(new_sentences, total_examples=len(new_sentences),epochs=basemodel.iter)
    average_word_vectors = sum(basemodel[element] for element in sentence.split())/len(sentence.split())        
    return(average_word_vectors)
    
def cos_sim(a, b):
	"""Takes 2 vectors a, b and returns the cosine similarity according 
	to the definition of the dot product
	"""
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)

tmp1 = word_cos_distance(1,"listen")
tmp2 = word_cos_distance(4,"listen")
print("Author1 and author4 listen cos distance")
print(cos_sim(tmp1, tmp2))

################################################################################
#Same thing considering for average vector
tmp1 = sentence_cos_distance(1,"her lips were parted")
tmp2 = sentence_cos_distance(4,"her lips were parted")
print("Author1 and author4 her lips were parted cos distance")
print(cos_sim(tmp1, tmp2))
################################################################################
#Now compare the result with the google's pretrained set
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
listen_tmp = model["listen"]
tmp1 = word_cos_distance(1,"listen")
tmp2 = word_cos_distance(4,"listen")
print("Author1 and google vector listen cos distance")
print(cos_sim(tmp1, listen_tmp))
print("Author4 and google vector listen cos distance")
print(cos_sim(tmp2, listen_tmp))

tmp1 = sentence_cos_distance(1,"her lips were parted")
tmp2 = sentence_cos_distance(4,"her lips were parted")
sentence = "her lips were parted"
average_word_vectors = sum(model[element] for element in sentence.split())/len(sentence.split())
print("Author1 and google vector her lips were parted cos distance")
print(cos_sim(tmp1, listen_tmp))
print("Author4 and google vector her lips were parted cos distance")
print(cos_sim(tmp2, listen_tmp))