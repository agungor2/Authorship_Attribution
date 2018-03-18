# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:08:24 2018

@author: mgungor
"""

import spacy

nlp = spacy.load('en')
doc1 = nlp('I had a good run')
print(doc1.vector)

doc2 = nlp('I had a great sprint')
print(doc2.vector)
#compare with the first sentence
print(doc2.similarity(doc1))


###############################################################################
#find the set of antonyms and synonyms given a word
from nltk.corpus import wordnet
synonyms = []
antonyms = []

for syn in wordnet.synsets("sprint"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))