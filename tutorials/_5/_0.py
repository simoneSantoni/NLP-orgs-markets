# !/usr/bin/env python3

"""
Docstring
--------------------------------------------------------------------------------
    _0.py    |    manipulate and represent word vectors with Numpy and Scipy
--------------------------------------------------------------------------------

Author: Simone Santoni, simone.santoni.1@city.ac.uk

Notes: This tutorial illustrates how:

       1. to load word vectors from a pre-trained model

       2. to fetch the word vectors associated with target entities

       3. to sample 'alter' words, i.e., words that are in the neighborhood
          of a target vector

       4. to fetc the word vectors associated with alter words

       5. to use dimensionality reduction techniques to explore the semantic
          similarity across target words
"""

# %% load libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
import seaborn as sns
import gensim.downloader as api
from gensim.models import Word2Vec
import pandas as pd


# %% download a pre-trained model of the language
'''
We will fetch the Word2Vec model trained on part of the Google News dataset.

!!! This is a large file, ~ 1.7 GB in memory ¡¡¡
'''

# %% load word2vec vectors
wv = api.load('word2vec-google-news-300')


# %% let's play a little bit with word vectors

# sample of american artists active in 'pop' music
artists = ['miley_cyrus' ,'beyonce', 'justin_bieber', 'chris_brown', 
           'drake', 'jayz', 'katy_perry', 'lady_gaga', 'madonna',
           'taylor_swift']

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


# retrieving words in the neighborhood of each artist

# --+ empty containers
embedding_clusters = []
word_clusters = []

# --+ iterate over artists
for artist in artists:
    # temporary lists
    embeddings = []
    words = []
    for similar_word, _ in wv.most_similar(artist, topn=30):
        # retrieve words
        words.append(similar_word)
        # retrieve vectors
        embeddings.append(wv[similar_word])
    # append temporary lists
    embedding_clusters.append(embeddings)
    word_clusters.append(words)


# %% dimensionality reduction applications

# initialize t-SNE
tsne_model = TSNE(n_components=2, init='pca')

# get coordinates for each artist
coordinates = tsne_model.fit_transform(vectors)

# arrange data in a Pandas df
df = pd.DataFrame({'x': [x for x in coordinates[:, 0]],
                   'y': [y for y in coordinates[:, 1]],
                   'artist': artists})

# plot data
# --+ create figure
fig = plt.figure()
# --+ add plot
ax = fig.add_subplot(1, 1, 1)
# --+ plot data
plot = ax.scatter(df.x, df.y, c=df.similarity, cmap='Reds')
for i in range(len(df)):
    ax.annotate("{}".format(df.artist[i].title(), df.similarity[i]),
                (df.x[i], df.y[i]))

plt.colorbar(mappable=plot, ax=ax)
plt.title('t-SNE visualization for {}'.format(search_word))
plt.show()

model = Word2Vec.load("GOT-vectors_300.w2v")
tsne_scatterplot(model, "bread")
