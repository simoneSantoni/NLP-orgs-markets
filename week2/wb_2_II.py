#!/usr/env/bin python3
#-*-encoding utf-8 -*-
"""
------------------------------------------------------------------------------
    wb_2_I.py    |    Python script for session 2 webinar, part I
------------------------------------------------------------------------------

Author     : Simone Santoni, simone.santoni.1@city.ac.uk 

Credits to : Bird, Klein & Loper (2009). Natural Language Processing with
             Python
            

Synopsis   : the script covers the following points:

             A. processign text with pure Python
             B. passing data through spaCy:
             C. manipulating data with NumPy/Scipy

TODO       : None

"""

# %% load libraries
import os
import glob
import numpy as np
import scipy as sc
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_lg")

# %% reading data
# navigate to data folder
os.getcwd()
os.chdir('../data/commencementSpeeches')
# read metadata
md = pd.read_csv('metadata.csv')
# read corpus of data
# --+ list of .txt files
in_fs = glob.glob(os.path.join('corpus/', '_*.txt'))
# --+ arrange data in a dictionary
d = {}
for in_f in in_fs:
    with open(in_f, 'r') as pipe:
        content = pipe.read()
    d[in_f] = content


# %% - A - processing raw text with pure Python
#  bare-bone tokenizer
tokens = d['corpus/_8.txt'].split(' ')
   
# %% - B - processing data through spaCy
# data transformation
doc = nlp(d['corpus/_8.txt'])
# let's visually inspect the outcome of the pipeline
for token in doc[0:10]:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)
# one possible application: reducing the variance in our corpus 
#   by retaining lemmas instead of words
lemmas = [token.lemma_ for token in doc[0:10]]

# %% - C - manipulating data with NumPy/Scipy
for token in doc[0:2]:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov, token.vector)
