# !/usr/bin/env python3
# -*- encoding utf-8 -*-
"""
--------------------------------------------------------------------------------
    semantic_similarity.py    |    fetch and manipulate word vectors with 
                              |        to measusre semantic similarity
--------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk

Synopsis : This tutorial pursues four learning goals:

            1. loading word vectors from trained embeddings
            2. retrieving word vectors associated with target entities
            3. visualizing clusters of words that are semantically similar
            4. calculating the distance between any two vectors (that is, their
               semantic (dis-)similarity)

To do    : None

"""

# %%
# load libraries
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import gensim.downloader as api

# %%
# we load a pre-trained model of the language available in Gensim
"""
We will fetch the Word2Vec model trained on part of the Google News dataset.

!!! This is a large file, ~ 1.7 GB in memory ¡¡¡
"""
wv = api.load("word2vec-google-news-300")

# %%
# let's play a little bit with word vectors

# sample of 'pop' singers
artists = [
    "taylor_swift",
    "beyonce",
    "alicia_keys",
    "katy_perry",
    "mariah_carey",
]
# retrieving word vectors
# --+ empty container
vectors = []
# --+ iterate over artists
for artist in artists:
    try:
        artis_vector = wv[artist]
        vectors.append(artis_vector)
    except:
        print("vector not available for {}".format(artist))


# %%
# let's measure the semantic distance between any pairs of entities/vectors

"""
basically, a quick & dirty approach to explore how the linguistic 
representations of artists are associated
"""
# empty conatiner to populate in the for loop
cs = np.empty(np.repeat(len(artists), 2))
# iterate over the elements of vectors
for i in range(len(artists)):
    for j in range(len(artists)):
        cs[i, j] = cosine(vectors[i], vectors[j])


# %%
# let's visualize the semantic similarity among entities

# figure
fig = plt.figure()
# plot
ax = fig.add_subplot(111)
# using the matshow() function
caxes = ax.matshow(cs, interpolation="nearest", cmap="inferno")
fig.colorbar(caxes)
# axes
ax.set_xticks(np.arange(0, len(artists), 1))
ax.set_yticks(np.arange(0, len(artists), 1))
# labels
labels = []
for artist in artists:
    split = artist.split("_")
    split = [s.title() for s in split]
    labels.append(" ".join(split))
ax.set_xticklabels(labels, rotation="vertical")
ax.set_yticklabels(labels)
# show/save figure
plt.show()


# %%
# that's it
