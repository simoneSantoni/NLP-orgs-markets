# !/usr/bin/env python3
# -*- encoding utf-8 -*-
"""
--------------------------------------------------------------------------------
    _1.py    |    manipulate and represent word vectors with Numpy and Scipy
--------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk

Synopsis : This tutorial pursues four learning goals:

            1. loading word vectors from trained embeddings
            2. browsing word embeddings to sample clusters of words that are 
               semantically similar
            3. visualizing clusters of words that are semantically similar
            4. calculating the distance between any two vectors (that is, their
               semantic (dis-)similarity)

To do    : None

"""

# %% load libraries
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import gensim.downloader as api

# %% download a pre-trained model of the language
"""
We will fetch the Word2Vec model trained on part of the Google News dataset.

!!! This is a large file, ~ 1.7 GB in memory ¡¡¡
"""
wv = api.load("word2vec-google-news-300")

# %% let's play a little bit with word vectors

# sample of american artists active in 'pop' music
artists = [
    "miley_cyrus",
    "beyonce",
    "justin_bieber",
    "chris_brown",
    "drake",
    "jayz",
    "katy_perry",
    "lady_gaga",
    "madonna",
    "taylor_swift",
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
        pass


# %% let's use NumPy/Scipy to manipulate the retrieved word vectors

# getting the covariance matrix of word vectors associated with focal artists
"""
basically, a quick & dirty approach to explore how the linguistic 
representations of artists are associated
"""

# --+ cosine similarity matrix
# ----+ empty list
cs = []

# ----+ iterate over the elements of vectors
for i in range(len(artists)):
    for j in range(len(artists)):
        if i < j:
            c_ij = cosine(vectors[i], vectors[j])
            cs.append([i, j, c_ij])
        else:
            pass

# ----+ alternative
cs = np.empty(np.repeat(len(artists), 2))


for i in range(len(artists)):
    for j in range(len(artists)):
        cs[i, j] = cosine(vectors[i], vectors[j])

# --+ visual display
# ----+ figure
fig = plt.figure()
# ----+ plot
ax = fig.add_subplot(111)
# ----+ using the matshow() function
caxes = ax.matshow(cs, interpolation="nearest")
fig.colorbar(caxes)
# ----+ axes
ax.set_xticks(np.arange(0, len(artists), 1))
ax.set_yticks(np.arange(0, len(artists), 1))
# ----+ labels
labels = []
for artist in artists:
    split = artist.split("_")
    split = [s.title() for s in split]
    labels.append(" ".join(split))
ax.set_xticklabels(labels, rotation="vertical")
ax.set_yticklabels(labels)
# ----+ show/save figure
plt.show()
#plt.savefig("cv_pop_artists.pdf", bbox_inches="tight")
