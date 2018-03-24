# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 10:51:18 2018
Character Frequency
@author: mgungor
"""
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import scipy.io as sc
from sklearn.metrics import f1_score

train_df = pd.read_csv("train1.csv")

toEnglishDict = {}
srcStr = ['à','â','ä','å','æ','ç','è','é','ê','ë','ï','î','ñ','ô','ö','õ','ü','û','α','δ','ν','ο','π','ς','υ','ἶ']
dstStr = ['a','a','a','a','a','c','e','e','e','e','i','i','n','o','o','o','u','u','a','d','n','o','p','s','y','i']
for src,dst in zip(srcStr,dstStr):
    toEnglishDict[src] = dst
    
# function that converts all non english chars to their closest english char counterparts
def myunidecode(inString):
    outString = ''
    for ch in inString:
        if ch in toEnglishDict.keys():
            outString += toEnglishDict[ch]
        else:
            outString += ch
    return outString

charsDict = {}
for key in range(1,51):
    charsDict[key] = []
charsDict["all"] = []
tmp = 0    
for k, (sentence, author) in enumerate(zip(train_df.text,train_df.author)):
    decodedSentence = myunidecode(sentence.lower())
    chars = [char for char in decodedSentence]
    
    charsDict['all']  += chars
    charsDict[author] += chars
    tmp += 1
    print(tmp)
    
charEncoder = preprocessing.LabelEncoder()
charEncoder.fit(charsDict['all'])

charCounts_1 = np.histogram(charEncoder.transform(charsDict[1]),range(len(charEncoder.classes_)+1),density=True)[0]
charCounts_3 = np.histogram(charEncoder.transform(charsDict[3]),range(len(charEncoder.classes_)+1),density=True)[0]
charCounts_8 = np.histogram(charEncoder.transform(charsDict[8]),range(len(charEncoder.classes_)+1),density=True)[0]

# sort the char classes by their usage frequency
sortedChars = np.flipud(np.argsort(charCounts_1 + charCounts_3 + charCounts_8))

barWidth = 0.21
x = np.arange(len(charCounts_1))

plt.figure(figsize=(12,7)); plt.title('Character Usage Frequncy - $P(C_t)$ ',fontsize=25);
plt.bar(x-barWidth, charCounts_1[sortedChars], barWidth, color='r', label='Arthur Doyle');
plt.bar(x         , charCounts_3[sortedChars], barWidth, color='g', label='Charles Dickens');
plt.bar(x+barWidth, charCounts_8[sortedChars], barWidth, color='b', label='James Baldwin');
plt.legend(fontsize=24); plt.ylabel('Usage Frequncy - $P(C_t)$', fontsize=20); plt.xlabel('$C_t$');
plt.xticks(x,["'%s'" %(charEncoder.classes_[i]) for i in sortedChars], fontsize=13);

prior_prob = {}
for i in train_df.author:
    prior_prob[i] = len(train_df[train_df.author==i])/float(len(train_df))
authorsList = np.unique(train_df.author.values)
authorsList = authorsList.tolist()
#Read Test data
test_df = pd.read_csv("test1.csv")
all_prob = np.zeros((len(test_df.text), 50))
all_prob_prior = np.zeros((len(test_df.text), 50))
predicted_author_id = np.zeros((len(test_df.text), 1))
predicted_author_prior_id = np.zeros((len(test_df.text), 1))


#character counts for all authors 
charCounts_all = {}
for i in authorsList:
    charCounts_all[i] = np.histogram(charEncoder.transform(charsDict[i]),range(len(charEncoder.classes_)+1),density=True)[0]
    print(i)

for k in range(len(test_df.text)):
    sentence = test_df.text[k]
    chars = [char for char in myunidecode(sentence.lower())]
    # convert to log so we can sum probabilities instead of multiply
    #Author wise calculate the probablity
    for i in authorsList:
        all_prob[k][i-1] = sum([np.log(charCounts_all[i][charEncoder.classes_ == ch]) for ch in chars])
        #consider prior probablity as well
        all_prob_prior[k][i-1] = tmp*prior_prob[i]
    
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