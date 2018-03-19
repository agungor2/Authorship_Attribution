# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 19:06:30 2018

@author: mgungor
"""

from gensim.models import Word2Vec
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
df = pd.read_csv("train.csv", encoding = "ISO-8859-1")


def display_closestwords_tsnescatterplot(model, word):
    
    arr = np.empty((0,300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
################################################################################
#Preperation of the text data for word2vec training
sentences = df[df.author == 1]["text"].ravel().tolist()
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
basemodel.train(new_sentences, total_examples=len(new_sentences))


display_closestwords_tsnescatterplot(basemodel, 'listen')
###############################################################################
#For author 4
sentences = df[df.author == 4]["text"].ravel().tolist()
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
basemodel.train(new_sentences, total_examples=len(new_sentences))


display_closestwords_tsnescatterplot(basemodel, 'listen')
