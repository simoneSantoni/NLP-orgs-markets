# !/usr/bin/env python3
# -*- encoding utf-8 -*-
"""
--------------------------------------------------------------------------------
    neighbor_vectors.py    |    retrieve a focal word's 'neighbor' vectors 
--------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk

Synopsis : This tutorial illustrates how:
   
           1. loading word vectors from a pre-trained model
           2. fetching the word vectors associated with target entities
           3. sampling 'alter' words, i.e., words that are in the neighborhood
              of a target vector
           4. fetching the word vectors associated with alter words
           5. using dimensionality reduction techniques to explore the semantic
              similarity across target and alter words

To do    : None

"""

# %%
# load libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
import seaborn as sns
import gensim.downloader as api
from gensim.models import Word2Vec
import pandas as pd

# %%
# custom colors
base_c = [i / 255 for i in [153, 0, 0]]
tri_1_c = [i / 255 for i in [25, 196, 49]]
tri_2_c = [i / 255 for i in [49, 25, 196]]

# %%
# download/load a pre-trained model of the language
"""
We will fetch the Word2Vec model trained on part of the Google News dataset.

!!! This is a large file, ~ 1.7 GB in memory ¡¡¡
"""
wv = api.load("word2vec-google-news-300")


# %%
# let's retrieve the focal words' vectors

# sample of super-star professional sport players
players = ["cristiano_ronaldo", "kobe_bryant", "tom_brady"]
# retrieving word vectors
# --+ empty container
vectors = []
# --+ iterate over players
for player in players:
    try:
        artis_vector = wv[player]
        vectors.append(artis_vector)
    except:
        pass


# %%
# let's retrieve the 10 closest vectors associated with each individual
# target word (i.e., sport super-start)

# empty containers
embedding_clusters = []
word_clusters = []
# iterate over players
for player in players:
    # temporary lists
    embeddings = []
    words = []
    for similar_word, _ in wv.most_similar(player, topn=10):
        # retrieve words
        words.append(similar_word)
        # retrieve vectors
        embeddings.append(wv[similar_word])
    # append temporary lists
    embedding_clusters.append(embeddings)
    word_clusters.append(words)

# %%
# plotting the position of the three super-starts in the vector space

# initialize t-SNE
tsne_model = TSNE(n_components=2)
# get coordinates for each player
coordinates = tsne_model.fit_transform(vectors)
# arrange data in a Pandas df
df = pd.DataFrame(
    {
        "x": [x for x in coordinates[:, 0]],
        "y": [y for y in coordinates[:, 1]],
        "player": players,
    }
)
# create figure
fig = plt.figure(figsize=(5, 5))
# add plot
ax = fig.add_subplot(1, 1, 1)
# plot data
plot = ax.scatter(df.x, df.y, marker="o", color=base_c, alpha=0.5)
# labels
labels = []
for player in players:
    split = player.split("_")
    split = [s.title() for s in split]
    labels.append(" ".join(split))
for i in range(len(df)):
    ax.annotate("{}".format(labels[i]), (df.x[i], df.y[i] + 10))
# hide all spines while preserving ticks
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
# axes
ax.set_xlabel(u"$D1$")
ax.set_ylabel(u"$D2$")
# grid
ax.grid(True, linestyle="--", color="grey", alpha=0.5)
# show/save figure
plt.show()


# %%
# plotting the positions of the neighbor words

# initialize r-SNE
tsne_model_en_2d = TSNE(
    perplexity=15, n_components=2, init="pca", n_iter=3500, random_state=32
)
# get np array out of alter words
embedding_clusters = np.array(embedding_clusters)
# --+ current shape
n, m, k = embedding_clusters.shape
# --+ reshape around players
embedding_clusters = embedding_clusters.reshape(n * m, k)
# coordinates to plot
tsne_output = tsne_model_en_2d.fit_transform(embedding_clusters)
# --+ coordinates to plot nested within players
embeddings_en_2d = np.array(tsne_output).reshape(n, m, 2)
# create figure
fig = plt.figure(figsize=(16, 8))
# add plot
ax = fig.add_subplot(1, 1, 1)
# colors
colors = [base_c, tri_1_c, tri_2_c]
# plot data
for label, embeddings, words, color in zip(
    labels, embeddings_en_2d, word_clusters, colors
):
    x = embeddings[:, 0]
    y = embeddings[:, 1]
    ax.scatter(x, y, c=color, label=label, alpha=0.5)
    for i, word in enumerate(words):
        plt.annotate(
            word,
            alpha=0.5,
            xy=(x[i], y[i]),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            size=8,
        )
# hide all spines while preserving ticks
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
# axes
ax.set_xlabel(u"$D1$")
ax.set_ylabel(u"$D2$")
# legend
plt.legend(loc="best")
# grid
plt.grid(True, linestyle="--", alpha=0.5)
# show/save figure
plt.show()


# %%
# that's it
