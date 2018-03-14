# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 18:40:23 2018

@author: mgungor
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS


# Read the whole text.
text = open('horse.txt').read()

horse_mask = np.array(Image.open('horse.jpg'))

stopwords = set(STOPWORDS)
stopwords.add("said")

wc = WordCloud(background_color="white", max_words=600, mask=horse_mask,
               stopwords=stopwords)
# generate word cloud
wc.generate(text)

# store to file
wc.to_file("horse_wordcloud.png")

# change the figure size
fig2 = plt.figure(figsize = (15,15)) # create a 20 * 20  figure 
ax3 = fig2.add_subplot(111)
ax3.imshow(wc, interpolation='bilinear')
ax3.axis("off")

#########################################################################################
#Same steps for Oliver Twist Charles Dickens

text = open('oliver.txt').read()

oliver_mask = np.array(Image.open('oliver.jpg'))

stopwords = set(STOPWORDS)
stopwords.add("said")

wc = WordCloud(background_color="white", max_words=100, mask=oliver_mask,
               stopwords=stopwords)
# generate word cloud
wc.generate(text)

# store to file
wc.to_file("oliver_wordcloud.png")

# show
fig2 = plt.figure(figsize = (15,15)) # create a 20 * 20  figure 
ax3 = fig2.add_subplot(111)
ax3.imshow(wc, interpolation='bilinear')
ax3.axis("off")
