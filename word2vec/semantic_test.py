# !/usr/bin/env python3
# -*- encoding utf-8 -*-
"""
------------------------------------------------------------------------------
    semantic_test.py    |    testing semantic expectations with word vectors
------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk

Synopsis : This tutorial pursues the following leaning goal:
   
           1. testing expectations about the semantic associations that tie 
              words together

To do    : None

"""

# %%
# load libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import spacy


# %%
#
my_words = ["doctor", "father", "mother", "nurse"]

# %%
# load the model
nlp = spacy.load("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")
word_embeddings = {}
for item in my_words:
    word_embeddings[item] = nlp.vocab[item].vector
X = np.array([word_embeddings[item] for item in my_words])
X_embedded = TSNE(
    n_components=2, learning_rate="auto", init="random"
    ).fit_transform(X)

# %%
# sample algebric operations
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1)
for item, coordinates in zip(my_words, X_embedded):
    ax.scatter(coordinates[0], coordinates[1], color="k")
    ax.annotate(item, (coordinates[0] + 2, coordinates[1] + 2))
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")  # increase margins ax.margins(0.4)
ax.set_xticks([])
ax.set_yticks([])
plt.show()

# %%
# that's it
