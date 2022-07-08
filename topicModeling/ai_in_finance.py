#!/usr/bin/env python3
# -*-encoding utf-8 -*-
"""
------------------------------------------------------------------------------
    ai_in_finance.py    |    LDA on an economic newspaper corpus
------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk

Synopsis : The corpus oftext contains documents published over circa 10 years.
           This LDA model analyzes the documents as a pooled cross section model
           (econometrics guys will appreciate that). The results of this script are
           included in Lanzolla, Gianvito, Simone Santoni, and Christopher Tucci.
           "Unlocking value from AI in financial services: strategic and organizational
           tradeoffs vs. media narratives." In Artificial Intelligence for Sustainable
           Value Creation, pp. 70-97. Edward Elgar Publishing, 2021.

Notes    : the code has been tested with Gensim 3.8.3, spaCy XX, and 
           Tomotpy xx.xx.

"""

# %%
# load libraries
import os
from tracemalloc import stop
from langcodes import best_match
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import spacy
import tomotopy as tp
from gensim.models import Phrases

# %%
# plot style
plt.style.use("Solarize_Light2")

# %%
# working dir
os.chdir("../sampleData/econNewspaper")

# %%
# load data

# load corpus
df = pd.read_csv("ft_wsj.csv")

# %%
# clean the data

# basic cleaning
# --+ date as date time
df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"])
# --+ get timespans
df.loc[:, "year"] = df["date"].dt.year
# --+ slice the data
"""
let's focus on the 2013 - 2019 timespan, which concentrates the large majority
of the data.
"""
df = df.loc[df["year"] >= 2013]
# --+ drop column
df.drop(["_id"], axis=1, inplace=True)
# --+ remove line breaks and unicode stuff
to_remove = ["\n", "\x19s", "\x1c"]
for item in to_remove:
    df.loc[:, "text"] = df.loc[:, "text"].str.replace(item, "")

# %%
# pre-process the data w/spaCy

# the spaCy pipeline
nlp = spacy.load("en_core_web_lg")
# expand the list of stopwords
stopwords = ["Mr", "Mr.", "$", "Inc.", "year"]
for item in stopwords:
    nlp.vocab[item].is_stop = True
# empty containers
docs_tokens = []
# iterate over the documents
for doc in nlp.pipe(
    df.loc[:, "text"].to_list(),
    batch_size=1000,
    disable=["tok2vec", "tagger", "parser", "attribute_ruler"],
):
    docs_tokens.append(
        [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and not token.like_num
        ]
    )

# %%
# find bigrams and trigrams

# get rid of common terms
common_terms = [
    u"of",
    u"with",
    u"without",
    u"and",
    u"or",
    u"the",
    u"a",
    u"not",
    "be",
    u"to",
    u"this",
    u"who",
    u"in",
]
# find phrases
bigram = Phrases(
    docs_tokens,
    min_count=50,
    threshold=5,
    max_vocab_size=50000,
    common_terms=common_terms,
)
trigram = Phrases(
    bigram[docs_tokens],
    min_count=50,
    threshold=5,
    max_vocab_size=50000,
    common_terms=common_terms,
)
# uncomment if bi-grammed, tokenized document is preferred
# docs_phrased = [bigram[line] for line in docs_tokens]
docs_phrased = [trigram[bigram[line]] for line in docs_tokens]

# %%
# create a corpus using Tomotopy utils

# empty corpus
corpus = tp.utils.Corpus()
# populate the corpus
for item in docs_phrased:
    corpus.add_doc(words=item)

# %%
# topic modeling ― explore model validity

# register "UMass" coherence scores
cvs = {}
for topic_number in range(1, 16, 1):
    mdl = tp.LDAModel(k=topic_number, corpus=corpus)
    for i in range(0, 100, 10):
        mdl.train(10)
        print("Iteration: {}\tLog-likelihood: {}".format(i, mdl.ll_per_word))
    coh = tp.coherence.Coherence(mdl, coherence="u_mass")
    cvs[topic_number] = coh.get_score()
# plot coherence scores
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
ax.plot(cvs.keys(), cvs.values(), "o-")
ax.set_xlabel("Number of topics retained")
ax.set_ylabel("Coherence score")
ax.set_xticks(range(1, 16, 1))
plt.show()

# %%
# topic model estimation

# let's go with the six topic model
best_mdl = tp.LDAModel(k=11, corpus=corpus)
for i in range(0, 100, 10):
    best_mdl.train(10)
    print("Iteration: {}\tLog-likelihood: {}".format(i, best_mdl.ll_per_word))

# %%
# word to topic probabilities

# an empty Pandas DF to populate
wt = pd.DataFrame()
# get word probabilities for each topic
for k in range(best_mdl.k):
    words, probs = [], []
    for word, prob in best_mdl.get_topic_words(k):
        words.append(word)
        probs.append(prob)
    tmp = pd.DataFrame(
        {
            "word": words,
            "prob": np.round(probs, 3),
            "k": np.repeat(k, len(words)),
            "sort": np.arange(0, len(words)),
        }
    )
    wt = pd.concat([wt, tmp], ignore_index=False)
    del tmp

# %%
# doc to topic probabilities

# get topic probabilities for each document
td = pd.DataFrame(
    np.stack([doc.get_topic_dist() for doc in best_mdl.docs]),
    columns=["topic_{}".format(i + 1) for i in range(best_mdl.k)],
)


# %%
