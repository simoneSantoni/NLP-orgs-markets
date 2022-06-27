#!/usr/env/bin python3
# -*- encoding utf-8 -*-
"""
------------------------------------------------------------------------------
    barebone_tomotopy.py    |    topic modeling with LDA and Tomotopy
------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk

Synopsis : The script shows how to implement LDA Topic Modeling with Tomotopy

Notes    : None

"""

# %%
# load libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
import tomotopy as tp
from rich.console import Console
from rich.table import Table

# %%
# working dir
os.chdir("../sampleData/tripadvisorReviews")

# %%
# read corpus
in_f = "hotel_reviews.csv"
df = pd.read_csv((in_f))

# %%
# pass the corpus through a spaCy pipeline

# initialize a pipeline
nlp = spacy.load("en_core_web_sm")
# process data
docs_tokens, tmp_tokens = [], []
for item in df.loc[:, "Review"].to_list():
    tmp_tokens = [
        token.lemma_
        for token in nlp(item)
        if not token.is_stop and not token.is_punct and not token.like_num
    ]
    docs_tokens.append(tmp_tokens)
    tmp_tokens = []

# %%
# Tomotopy LDA estimation

# create a corpus using tp utilities
corpus = tp.utils.Corpus()
# populate the corpus
for item in docs_tokens:
    corpus.add_doc(words=item)
# estimate a model with 10 topics
lda = tp.LDAModel(k=10, corpus=corpus)
# train the model
for i in range(0, 100, 10):
    lda.train(10)
    print("Iteration: {}\tLog-likelihood: {}".format(i, lda.ll_per_word))

# %%
# inspect the output of the LDA algorithm

# create a Rich's table to print the output of the spaCy's pipeline
console = Console()
# defin table properties
table = Table(
    show_header=True,
    header_style="cyan",
    title="[bold] [cyan] Word to topic probabilities (top 10 words)[/cyan]",
    width=150,
)
# add columns
table.add_column("Topic", justify="center", style="cyan", width=10)
table.add_column("W 1", width=12)
table.add_column("W 2", width=12)
table.add_column("W 3", width=12)
table.add_column("W 4", width=12)
table.add_column("W 5", width=12)
table.add_column("W 6", width=12)
table.add_column("W 7", width=12)
table.add_column("W 8", width=12)
table.add_column("W 9", width=12)
table.add_column("W 10", width=12)
# add rows
for k in range(lda.k):
    values = []
    for word, prob in lda.get_topic_words(k):
        values.append("{}\n({})\n".format(word, str(np.round(prob, 3))))       
    table.add_row(
        str(k),
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
        values[9],
    )
# print the table
table

# %%
# get the coherence score of the model

# register the coherence of the estimated model
coh = tp.coherence.Coherence(lda, coherence="u_mass")
# model coherence
average_coherence = coh.get_score()
# topic coherence
coherence_per_topic = [coh.get_score(topic_id=k) for k in range(lda.k)]
# plot results
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(range(lda.k), coherence_per_topic)
ax.set_xticks(range(lda.k))
ax.set_xlabel("Topic number")
ax.set_ylabel("Coherence score")
plt.axhline(y=average_coherence, color="orange", linestyle="--")
plt.show()

# %%
# that's it
