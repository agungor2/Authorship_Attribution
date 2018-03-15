# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:20:04 2018

@author: mgungor
"""

import scipy.io as sc
import pandas as pd
import csv
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

#Create a data frame for the dataset
columns = ["id","text","author"]
all_data = []
with open('dataset_with_stop_words.csv', 'w') as file:
    writer = csv.writer(file, delimiter = ',', lineterminator = '\n')
    writer.writerow(columns)
    for i in range(len(txt_pieces)):
        tmp_txt = ''
        for j in range(len(txt_pieces[i,:])):
            tmp_ind = txt_pieces[i,j]
            if tmp_ind ==0:
                break
            else:
                tmp_txt = tmp_txt + shortened_vocab[0][tmp_ind-1][0] + ' '
        writer.writerow([str(i), tmp_txt, str(aid[i][0])])
        print(i)