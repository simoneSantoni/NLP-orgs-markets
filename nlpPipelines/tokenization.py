# !/usr/env/bin python3
# encoding -*- utf-8 -

"""
------------------------------------------------------------------------------
    tokenization.py    |    Three tokenizers in action
------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk 

Synopsys : The script implments a barebone tokenizer in pure Python. Sample 
           data from IMDB are used to illustrate the functioning of the 
	   tokenizer.

Notes    : NaN 

"""

# %% Import libraries
import glob, os
import json
import numpy as np
import nltk

nltk.download("punkt")  ## comment this if you have already downloaded 'punkt'
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import sent_tokenize
import spacy
nlp = spacy.load("en_core_web_lg")

# %% Browse dirs and sub-dirs to locate movie reviews
# specifiy the target folder
target_folder = "../sampleData/movieReviews"
# retrieve the full pathways pointing to individual files
in_fs = glob.glob(os.path.join(target_folder, "*.json"))

# %% Read files
# create a container for the files
reviews = {}
for in_f in in_fs:
    # open the file
    with open(in_f, "r") as f:
        # read the file
        text = json.load(f)
        # upate the container
        key = in_f.split("movieReviews/")[1].rstrip(".json")
        reviews[key] = text
# let's tokenize a randomly drawn review
# --+ reviews keys
keys = list(reviews.keys())
# --+ we randomly pick up a key
pos = np.random.choice(len(keys))
text = reviews[keys[pos]]["imdb_user_review"]

# %% Bare-bone tokenizer
# get tokens by splitting on spaces
tkns_bb = text.split(" ")
print(tkns_bb)

# %% Tokenization with NLTK
"""
NLTK implements a variety of tokenizers, including:

- `wordpunct_tokenize', a tokenizer that uses regular expressions to breaks a 
  piece of text down based on whitespaces and punctuation 
- `word_tokenize', a tokenizer leveraging the punkt unsupervised algorithm to
   build a model for abbreviation words, collocations, and words that start
   sentences
- `sent_tokenize', a tokenizer that uses regular expressions to break a text 
  down into sentences
"""

# regular expression-based tokenizer
tkns_wp = wordpunct_tokenize(text)
print(tkns_wp)

# punkt-based tokenizer
tkns_pu = word_tokenize(text)
print(tkns_pu)
# sentence tokenizer
snt_tkns = [[sentence] for sentence in sent_tokenize(text)]
print(snt_tkns)

# %% Tokenization with spaCy
tkns_sp = [token.text for token in nlp(text)]
