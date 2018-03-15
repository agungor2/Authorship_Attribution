# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 06:35:38 2017

@author: mgungor


Finding the unmatched words and replacing them for data set with stop words included
"""

import scipy.io as sc
import gensim
import pandas as pd
import numpy as np

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

authorship_attribution = sc.loadmat("ml_challenge_data_wstopwords.mat")
vocabulary = sc.loadmat("Vocabulary_wstopwords.mat")

#Split the categories

aid = authorship_attribution['aid']
bid = authorship_attribution['bid']
ind = authorship_attribution['ind']
WW  = authorship_attribution['WW']
test_ind = authorship_attribution['test_ind']
train_ind = authorship_attribution['train_ind']
txt_pieces = authorship_attribution['txt_pieces']


##
shortened_vocab = vocabulary['shortened_vocab']
tfidf = vocabulary['tfidf']
vocab = vocabulary['vocab']



Ml_word2vec=[]
store_unused=[]
store_index=[]
count=0
for i in range(0,len(shortened_vocab[0])):
    
    if(model.vocab.get(shortened_vocab[0][i][0],0)==0):
        Ml_word2vec.append(np.zeros(300))
        count=count+1
        store_unused.append(shortened_vocab[0][i][0])
        store_index.append(i+1)
        print(shortened_vocab[0][i][0])
    else:
        Ml_word2vec.append(model[shortened_vocab[0][i][0]])
        
        
        
reusable = [ 'theaters',
 'humored',
 'program',
 'endeavoring',
 'coloring',
 'endeavored',
 'honors',
 'harbor',
 'parlor',
 'equaled',
 'labors',
 'quarreling',
 'willful',
 'fiber',
 'splendor',
 'worshiped',
 'englishman',
 'quarreled',
 'behavior',
 'fulfillment',
 'catalog',
 'honored',
 'plow',
 'recognize',
 'favored',
 'tranquility',
 'neighboring',
 'fulfill',
 'endeavor',
 'favorable',
 'honorable',
 'colors',
 'learned',
 'recognized',
 'neighbors',
 'neighbor',
 'humor',
 'mold',
 'labor',
 'favorite',
 'neighborhood',
 'colored',
 'travelers',
 'pretense',
 'favor',
 'ax',
 'color',
 'honor',
 'gray',
 'marvelous',
 'traveler',
 'practiced',
 'theater',
 'traveled',
 'defense',
 'offense',
 'traveling',
 'center']

#Matlab index, need to subtract 1
        
reusable_index =[ 39,
 42,
 81,
 205,
 313,
 542,
 793,
 873,
 1231,
 1393,
 1608,
 1617,
 1626,
 1646,
 1688,
 1773,
 1832,
 1879,
 1918,
 1977,
 2027,
 2055,
 2079,
 2218,
 2253,
 2310,
 2447,
 2641,
 2902,
 2948,
 3513,
 3539,
 3697,
 3783,
 4026,
 4117,
 4287,
 4326,
 4373,
 4409,
 4410,
 4908,
 5113,
 5124,
 5300,
 5324,
 5347,
 5363,
 5549,
 5607,
 5736,
 5794,
 6001,
 6689,
 6725,
 6745,
 7303,
 8406]

for i in range (0,len(reusable_index)):
    Ml_word2vec[int(reusable_index[i])-1]=model[reusable[i]]
                
df = pd.DataFrame(Ml_word2vec)
df.to_csv('word2vec_data_stop_words.csv', sep=',',header=False, index=False)


        
"""
Write a piece of the data 
"""
all_words = []
for i in range(0,10000):
    all_words.append(shortened_vocab[0][i][0])

tmp_txt = txt_pieces[1,:]

decrypt_txt = ''

for i in range(len(tmp_txt)):
    decrypt_txt = decrypt_txt + all_words[int(tmp_txt[i])-1]
    decrypt_txt = decrypt_txt + ' '
    
                                          
print(decrypt_txt)


"""
Check for the left over words
"""
count = 0
for i in range(len(Ml_word2vec)):
    if (Ml_word2vec[i]==np.zeros((1,300))).all():
        print shortened_vocab[0][i][0]
        count +=1
        
#New words to add in
reusable = {"drafts": 389,
 "rumor":548,
 "spoiled":590,
 "mustache":717,
 "practice":2790,
 "from":3314,
 "musn't": 3951,
 "draft":5749,
 "vigor":1924}        
        
for key in reusable:
    Ml_word2vec[reusable[key]] =model[key]
    print key