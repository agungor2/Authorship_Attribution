# Authorship Attribution
Benchmarking Authorship Attribution Techniques Using 1113 Books by Fifty Victorian Era Authors, Abdulmecit Gungor

The followings are possible techniques that can be used for text classification Authorship Attribution studies.
A dataset of 3 authors and 50 authors of Victorian era writers are collected and this repository is built to form as a tutorial and benchmarking for any dataset that's under Authorship Attribution Category. 


Sentiment_Analysis.ipynb :
It serves as an examplary case of sentiment analysis using nltk. Naive bayes classifier has been built to predict if a given text is positive or negative. 

Classification_MLP.ipynb :
MLP Classifier has been built to make a sentiment analyzer model by representing a text with it's word2vec average form. 

LSTM_classification.py : 
It's a simple example case usage of LSTM on IMDB text dataset. 

Google_NLP_api.py :
Using Google NLP Api to do sentiment analysis

Translated_list.txt : 
Translated book lists from German, French, Russian and mix languages of European authors

Clean_data.py : 
How to clean a raw text data and extract top 10000 most occuring words

BOW_50author.html :
BOW extractition and top 20 of them

Author_Distribution.py :
Author distribution and plotting a histogram of number of texts per author

oliver_horse.py :
Show the oliver twist and horse tale word usage frequency with wordcloud

bow_3authors.py :
Bag of words representation for 3 authors dataset

convert_data.py : 
Conversion of 50 authors dataset from numeric to word form

word2vec_extract_data.py :
Using word2vec to create a list of top 10,000 words representation in a matrix and showing British & American English different words in the database

n-grams.py :
N-grams extraction for 3 authors and 50 authors dataset

n-grams2.py :
top 20 n-grams extraction for 3 authors and 50 authors dataset

stylometric_features.py :
Stylometric features such as Number of punctuations density, Number of title case density, Average length of the words in the text, Number of punctuations in the text, and many more. It shows how to extract such features from  3 authors dataset. 

vocab_diversity.py :
Vocabulary richness measure for both 3 authors and 50 authors dataset. 

power_words.py :
Usage of power words authorwise

tfidf_example.py :
TF-IDF meaure calculation for both 50 authors and 3 authors dataset

moses_tokinezer.py :
Using moses tokenizer to find out "is, 's, n't, not" usage authorwise

char_ngram.py :
Character n-gram representation for both 3 authors and 50 authors dataset

syntactic_features.py : 
syntactic feature calculation authorwise (such as usage of adjectives, verbs, nouns, etc...)

P_N_index.py : 
Positivity and negativity index comparison per author in both 3 and 50 authors dataset

synonym_example.py : 
Synonym and antonym word usage examplary cases

word2vec_tsne.py : 
Train it on author 1 and author 4 and plot it in 2 dimensional vector space considering the word of your choosing
