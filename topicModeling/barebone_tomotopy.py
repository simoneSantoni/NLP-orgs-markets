#!/usr/env/bin python3
# -*- encoding utf-8 -*-
"""
------------------------------------------------------------------------------
    _7.py    |    topic modeling with Tomotopy
------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk

Synopsis : The script shows how to implement LDA Topic Modeling with Tomotopy

To do    : None

"""

# %%
# load libraries
import os
import pandas as pd
import spacy
import tomotopy as tp

# %%
# working dir
os.chdir("../sampleData/tripadvisorReviews")

# %% r
# ead corpus
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
# display words by topics
for k in range(lda.k):
    print("Top 10 words of topic #{}".format(k))
    print(lda.get_topic_words(k, top_n=10))
# save model estimates
lda.save("hotel_review_lda_estimates.bin")

# %%
import numpy as np
for k in range(lda.k):
        print('Topic #{}'.format(k))
        for word, prob in lda.get_topic_words(k):
            print('\t', word, np.round(prob, 3), sep='\t')
# %%
