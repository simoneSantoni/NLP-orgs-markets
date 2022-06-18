# !/usr/bin/env python3
# -*- encoding utf-8 -*-
"""
--------------------------------------------------------------------------------
    _4.py    |    analogical reasoning with word embeddings
--------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk

Synopsis : This tutorial pursues the following leaning goal:
   
           1. expanding on word embeddings to uncover complex semantic relations 
              involving lexical items

To do    : None

"""

# %% load libraries
from scipy import spatial
import spacy
import en_core_web_lg

# %% load the model
nlp = en_core_web_lg.load()

# %% 
'''
Let's apply algebric operations to show how embeddings
reveal complex semantic relations

rome + italy = paris + france
'''
# fetch the vectors associated with the four lexical items
rome = nlp.vocab['rome'].vector
italy = nlp.vocab['italy'].vector
paris = nlp.vocab['paris'].vector
france = nlp.vocab['france'].vector
 
# let's get the vector in the vocabulary that is most proximal 
# to 'rome' - 'paris' + 'france'
maybe_italy = rome - paris + france

# my similarity function
cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)

# get similar items
computed_similarities = []
for word in nlp.vocab:
    # Ignore words without vectors
    if not word.has_vector:
        continue
 
    similarity = cosine_similarity(maybe_italy, word.vector)
    computed_similarities.append((word, similarity))
 
computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])
print([w[0].text for w in computed_similarities[:10]])

"""
well, 'italy' is the second most closely associated item after 'rome'
""" 
