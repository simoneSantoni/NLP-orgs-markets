# !/usr/bin/env python3
# -*- encoding utf-8 -*-
"""
--------------------------------------------------------------------------------
    _3.py    |    manipulate and represent word vectors with Numpy and Scipy
--------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk

Synopsis : This tutorial pursues the following learning goals:
   
           1. charting the semantic relations involving sets of lexical items
           2. familiarizing with 'whatlies,' a dedicated libraries for the 
              visualization of word embeddings

To do    : None

"""

# %% load libraries
from whatlies import EmbeddingSet
from whatlies.language import SpacyLanguage

# %% load a model of the language 'via' whatlies
lang = SpacyLanguage("en_core_web_lg")

# %% create a list of lexical items
"""
let's see how animals (actors) map onto qualities (attributes)
"""
# sample animals
animals = ["cat", "dog", "mouse"]
# sample qualities
qualities = ["responsive", "loyal"]
# set of lexical items
items = animals + qualities

# %% browse the loaded model of the language, retrieve the vectors
#    and create and initialize an embedding sets (a class specific
#    to the library whatlies)
emb = EmbeddingSet(*[lang[item] for item in items])
# the position of animals should be relative to the word vectors
#   for 'smart' and 'loyal'
emb.plot_interactive(x_axis=emb["responsive"], y_axis=emb["loyal"])
