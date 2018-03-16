# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:15:03 2018

@author: mgungor
"""

import pandas as pd 
train_df = pd.read_csv("train.csv")
from sklearn.feature_extraction.text import TfidfVectorizer

###############################################################################
#Built it on the 3 authors dataset
for author_i in train_df.author.unique():
    print(author_i)
    at = train_df[train_df.author == author_i].text.values
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')
    tfidf_matrix =  tf.fit_transform(at)
    feature_names = tf.get_feature_names() 
    len(feature_names)
    dense = tfidf_matrix.todense()
    episode = dense[0].tolist()[0]
    phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
    sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
    for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:20]:
       print('{0: <20} {1}'.format(phrase, score))
    print("\n")
    
###############################################################################
#Built it on the 50 authors dataset
train_df = pd.read_csv("train.csv",encoding = "ISO-8859-1")
for author_i in train_df.author.unique():
    print(author_i)
    at = train_df[train_df.author == author_i].text.values
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')
    tfidf_matrix =  tf.fit_transform(at)
    feature_names = tf.get_feature_names() 
    len(feature_names)
    dense = tfidf_matrix.todense()
    episode = dense[0].tolist()[0]
    phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
    sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
    for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:20]:
       print('{0: <20} {1}'.format(phrase, score))
    print("\n")
    
"""
Some results for 3 authors dataset
EAP
afforded means       0.16323096985913788
afforded means ascertaining 0.16323096985913788
ascertaining dimensions 0.16323096985913788
ascertaining dimensions dungeon 0.16323096985913788
aware fact           0.16323096985913788
aware fact perfectly 0.16323096985913788
circuit return       0.16323096985913788
circuit return point 0.16323096985913788
dimensions dungeon   0.16323096985913788
dimensions dungeon make 0.16323096985913788
dungeon make         0.16323096985913788
dungeon make circuit 0.16323096985913788
fact perfectly       0.16323096985913788
fact perfectly uniform 0.16323096985913788
make circuit         0.16323096985913788
make circuit return  0.16323096985913788
means ascertaining   0.16323096985913788
means ascertaining dimensions 0.16323096985913788
perfectly uniform    0.16323096985913788
perfectly uniform wall 0.16323096985913788


HPL
fumbling mere        0.3618562880458831
fumbling mere mistake 0.3618562880458831
mere mistake         0.3618562880458831
occurred fumbling    0.3618562880458831
occurred fumbling mere 0.3618562880458831
mistake              0.3174075599798223
fumbling             0.30576823795531416
mere                 0.2861250889570534
occurred             0.2630414283016087


MWS
looked               0.1717852211792789
beneath speckled     0.14329695310915957
beneath speckled happy 0.14329695310915957
cheering fair        0.14329695310915957
cottages wealthier   0.14329695310915957
cottages wealthier towns 0.14329695310915957
counties spread      0.14329695310915957
counties spread beneath 0.14329695310915957
fertile counties     0.14329695310915957
fertile counties spread 0.14329695310915957
happy cottages       0.14329695310915957
happy cottages wealthier 0.14329695310915957
heart cheering       0.14329695310915957
heart cheering fair  0.14329695310915957
looked windsor       0.14329695310915957
looked windsor terrace 0.14329695310915957
looked years         0.14329695310915957
looked years heart   0.14329695310915957
lovely spring        0.14329695310915957
lovely spring looked 0.14329695310915957

"""

"""
Some result for 50 authors dataset
1
rowed                0.08398631405812844
canal                0.07308336161251015
time listen          0.06793864953475896
boat                 0.054238432180660406
effect               0.05217446908453025
craft                0.04752854288973964
city                 0.04728890384180398
row                  0.04506959105600333
passenger            0.04385042653211469
noise                0.04311643845086123
does                 0.04231273264743283
silently             0.040917902472524526
early                0.038693334312835365
stone                0.038458950638788215
corner               0.036865732755332226
lines                0.036330799763288756
love gone            0.0341680312884493
african forest       0.03396932476737948
african forest things 0.03396932476737948
agreeable best       0.03396932476737948


2
mr                   0.11934830210891315
mrs                  0.0770225186593897
sir john             0.07240954297329026
try nut              0.06499744591558604
fine lady            0.060784069554367326
old mr               0.060784069554367326
mr says              0.05779463145136934
don know             0.05636066221887423
lady                 0.05326132738332029
dr                   0.05288247971143559
nut                  0.050591816987152614
john                 0.050040597677046134
silence              0.048287381184243215
wrote                0.04701424961919152
henry                0.04637844062593388
don                  0.04096941466896419
says                 0.03961198926589446
sister               0.03961198926589446
sir                  0.03923637598072106
disagreeable         0.039175626161717166


3
priest               0.1278565155154648
women                0.08935697404181811
half                 0.0821985867247272
path                 0.0655858669643553
forest               0.06242876783456597
dead                 0.05956656217656841
old square           0.05949941332316239
old square church    0.05949941332316239
settlement           0.05949941332316239
square church        0.05949941332316239
boughs               0.05524669624682622
consecrated ground   0.05524669624682622
distant              0.05524669624682622
husbands             0.05524669624682622
ground               0.053830139839618325
grave                0.05225555318067016
consecrated          0.05222934546224255
stolid               0.05222934546224255
parson               0.04988890636607478
gathered             0.04635982039103612


4
george               0.1387755289507923
said george          0.09571581115012358
girl                 0.08688172932457675
stray                0.07317778516359498
hotel                0.06596357932167768
miss said george     0.06381054076674905
pump room            0.06381054076674905
drive                0.05996034775002795
pump                 0.057997941685139956
coach                0.05741964082345442
box seat             0.05459778919067241
flourish             0.05330511120665535
said girl            0.05031410365484442
miss                 0.049328747626640856
cargo                0.04586435709119513
draws                0.04586435709119513
seat                 0.04497026081252096
inevitable           0.04493164028497396
miss said            0.04332948777697932
going                0.037123158825598544


6
writer               0.0913483801238887
virtue               0.07829985365343989
tale                 0.07609659102970809
neighbours           0.07526428259347967
expensive            0.0707895858982278
tale humble          0.06706029819630069
obligation           0.05539966690851506
preface              0.05539966690851506
habits               0.05371219328310883
pecuniary            0.05234621974652027
humble               0.04811760649941571
envy                 0.046406477916534294
silently             0.046406477916534294
sufficiently         0.044989180521630055
country              0.044393673934642254
patiently            0.04434571296783096
success              0.041137314674258366
character            0.039462734325123996
fate                 0.03943544355783402
simple               0.037632141296739835


"""