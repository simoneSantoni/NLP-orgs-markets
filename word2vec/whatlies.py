# !/usr/bin/env python3
# -*- encoding utf-8 -*-
"""
--------------------------------------------------------------------------------
    whatlies.py    |    interactive viz of word vectors
--------------------------------------------------------------------------------

author   : simone santoni, simone.santoni.1@city.ac.uk

synopsis : this tutorial pursues the following learning goals:
   
           1. familiarizing with 'whatlies,' a dedicated libraries for the 
              visualization of word embeddings

to do    : none

"""

# %% 
# load libraries
from whatlies import EmbeddingSet
from whatlies.language import SpacyLanguage

# %% 
# load a model of the language 'via' whatlies
lang = SpacyLanguage("en_core_web_lg")

# %% 
# create a list of lexical items
"""
let's see how animals (actors) map onto qualities (attributes)
"""
# sample animals
animals = ["cow", "dog", "duck", "horse", "pig", "rooster", "sheep"]
# sample qualities
qualities = ["intelligent", "loyal"]
# set of lexical items
items = animals + qualities

# %% 
# create an interactive visualization to compare and contrast animals' 
# qualities

# create a set of embeddings for the animals and the two qualities
emb = EmbeddingSet(*[lang[item] for item in items])
# the position of animals should be relative to the qualities
emb.plot_interactive(x_axis=emb["intelligent"], y_axis=emb["loyal"])

# %%
# that's it