# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 08:19:46 2018

@author: Mecit
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

train_df = pd.read_csv("train.csv")


cnt_srs = train_df['author'].value_counts()


plt.figure(figsize=(8,4))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Author Name', fontsize=12)
plt.show()