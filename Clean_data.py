# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 15:55:29 2018

@author: Mecit
"""

import re
from bs4 import BeautifulSoup             
from nltk.corpus import stopwords
import glob
import pandas as pd
import operator
    

def Clean_data( raw_review ):
    #Clean dataset
    
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. Selact stop words and remove them (Optional)
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string
    return( " ".join( meaningful_words ))
    
path =r'rawdata_files' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_of_all_words = {}
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    for element in Clean_data(df)["texts"]:
        if element in list_of_all_words:
            list_of_all_words[element] += 1
        else:
            list_of_all_words[element] = 1
            
#Sort the list of all words to identify top 10,000 words
sorted_list_of_all_words = sorted(list_of_all_words.items(), key=operator.itemgetter(0))