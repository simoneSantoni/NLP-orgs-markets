#!/usr/env/bin python3
# -*- encoding utf-8 -*-
"""
------------------------------------------------------------------------------
    _8.py    |    implementation of the semantic axis method
------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk

Synopsis : The script shows how to implement LDA Topic Modeling with Tomotopy

To do    : None

"""

# %% load libraries
import os
import numpy as np
from scipy import spatial
import pandas as pd
import spacy
import en_core_web_lg
import tomotopy as tp

# %% initialize a spaCy's pipeline
nlp = en_core_web_lg.load()

# %% load General Inquirer categories of words
#    Note: docs available at http://www.wjh.harvard.edu/~inquirer/homecat.htm
df = pd.read_excel("./inquireraugmented.xls", usecols=["Entry", "Power", "Submit"])

# %% custom function that creates a semantic axis from a sample
#    of words coming from the General Inquirer
power = df.loc[df["Power"].notnull(), "Entry"].sample(n=10, random_state=123).to_list()
submit = df.loc[df["Submit"].notnull(), "Entry"].sample(n=10, random_state=123).to_list()
# cleaning
def clean(word_):
    return word_.lower().split("#")[0]


power = [clean(word) for word in power]
submit = [clean(word) for word in submit]

# %% get the affect score for a sample of unseen words
def semantic_ax(word_list, vector_len=300):
    wv = {}
    # step 2, get the word vectors
    for word in word_list:
        wv[word] = nlp.vocab[word].vector
    # step 3, get the centroid of each pole
    centroid = []
    for i in range(vector_len):
        dimension = [wv[word][i] for word in wv.keys()]
        centroid.extend([np.mean(dimension)])
    # return data
    return wv, np.array(centroid)


power_wv, power_centroid = semantic_ax(power)
submit_wv, submit_centroid = semantic_ax(submit)

# %% step 4, get the semantic axis
my_ax = power_centroid - submit_centroid

# %% step 5, appreciating the position of unseen words along the semantic axis
unseen = ['abolish', 'affliction']
for word in unseen:
    if nlp.vocab[word].vector is not None:
        pos = 1 - spatial.distance.cosine(my_ax, nlp.vocab[word].vector)
        print(
        """
        =============================================================
        Unseen word:                                  {}
    
        Position along the submit - power continuum:  {:.2f}
    
        ------------------------------------------------------------- 
        """.format(word, pos))
    else:
        print(
        """
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        Too bad: no word vector avialble for the unseen word {}
        
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """.format(word)
        )

# %%
