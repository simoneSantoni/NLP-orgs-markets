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

Notes    : the code has been tested with Gensim 3.8.3 and Mallet 2.0.8

"""

# %%
# load libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import spacy
import gensim
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import Phrases
from gensim.models.ldamodel import LdaModel, CoherenceModel
from gensim.similarities import MatrixSimilarity

# %%
# working dir
os.chdir("../sampleData/econNewspaper")

# %%
# load data

# load corpus
df = pd.read_csv("ft_wsj.csv")

# %%
# clean data read from Mongo

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

# %%
# pre-process the data w/spaCy

# the spaCy pipeline
nlp = spacy.load("en_core_web_sm")
# empty containers
docs_tokens, tmp_tokens = [], []
# iterate over the documents
for doc in df.text.to_list():
    tmp_tokens = [
        token.lemma_
        for token in nlp(doc)
        if not token.is_stop and not token.is_punct and not token.like_num
    ]
    docs_tokens.append(tmp_tokens)
    tmp_tokens = []

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
# create corpus and dictionary to pass to Gensim

# dictionary
dict = Dictionary(docs_phrased)
# vector represenation of the documents
corpus = [dict.doc2bow(doc) for doc in docs_phrased]


# %%
# topic modeling ― explore model validity

# define function to explore a gamut of competing models
def compute_coherence_values(
    _dictionary, _corpus, _texts, _limit, _start, _step,
):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    -----------
    dictionary : Gensim dictionary
    corpus     : Gensim corpus
    texts      : List of input texts
    limit      : Max number of topics

    Returns:
    --------
    model_list       : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model
                       with respective number of topics
    """
    # containers
    coherence_values = []
    model_list = []
    for num_topics in range(_start, _limit, _step):
        model = LdaModel(corpus=_corpus, num_topics=num_topics, id2word=_dictionary,)
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model, texts=_texts, dictionary=_dictionary, coherence="c_v"
        )
        coherence_values.append(coherencemodel.get_coherence())
    # return output
    return model_list, coherence_values


# %%
# collect coherence scores as the number of retained topics change

# 5 - 9 topic models
# --+ search grid
limit, start, step = 10, 5, 1
# --+ run function
ml_5_9, cv_5_9 = compute_coherence_values(
    _dictionary=dict,
    _corpus=corpus,
    _texts=docs_phrased,
    _start=start,
    _limit=limit,
    _step=step,
)

# %%
# plot collected coherence scores data
# --+ create figure
fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(6, 4), sharey=True)
# --+ data series
x0 = np.arange(5, 10, 1)
y0 = cv_5_9
x1 = np.arange(10, 35, 5)
y1 = cv_10_30
# --+ reference points
_min = np.min([y0, y1]).round(2)
_max = np.max([y0, y1]).round(2)
# --+ create PANEL A
# --+ plot data
ax0.plot(x0, y0, marker="o", color="k", ls="")
# axes
ax0.set_xlabel("Number of topics retained")
ax0.set_ylabel("Coherence score")
ax0.set_xticks(np.arange(5, 10, 1))
ax0.set_yticks(np.arange(_min, _max, 0.02))
# reference line
# ax0.ax0vline(x=11, ymin=0, ymax0=1, color='r')
# grid
ax0.grid(True, ls="--", axis="y", which="major")
# --+ hide all spines while preserving ticks
ax0.spines["right"].set_visible(False)
ax0.spines["top"].set_visible(False)
ax0.spines["bottom"].set_visible(False)
ax0.spines["left"].set_visible(False)
ax0.yaxis.set_ticks_position("left")
ax0.xaxis.set_ticks_position("bottom")
ax0.yaxis.set_ticks_position("left")
ax0.xaxis.set_ticks_position("bottom")
# -- textbox
ax0.text(5, 0.51, u"$A$", fontsize=13)
# --+ PANEL B
# --+ plot data
ax1.plot(x1, y1, marker="o", color="k", ls="")
# axes
ax1.set_xlabel("Number of topics retained")
ax1.set_xticks(np.arange(10, 35, 5))
# grid
ax1.grid(True, ls="--", axis="y", which="major")
# --+ hide all spines while preserving ticks
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.spines["left"].set_visible(False)
# ax1.yaxis.set_ticks_position('left')
# ax1.xaxis.set_ticks_position('bottom')
# ax1.yaxis.set_ticks_position('left')
# ax1.xaxis.set_ticks_position('bottom')
# --+ textbox
ax0.text(10, 0.51, u"$B$", fontsize=13)
# --+ write plot to file
out_f = os.path.join("analysis", "topicModeling", ".output", "pr_coherence_scores.pdf")
plt.savefig(out_f, transparent=True, bbox_inches="tight", pad_inches=0)


# %%
# topic model estimation
"""
I focus on two models:
    - 8 topics, ~ local optimum
    - 30 topic, ~ global optimum
"""

# model with 30 topics
# ----+ estimate model
_path = "/home/simone/.mallet/mallet-2.0.8/bin/mallet"
lda_30 = LdaModel(
    corpus=corpus, id2word=dict, num_topics=10
)
# ----+ print topics (20 words per topic)
lda_30.print_topics(num_topics=30, num_words=20)
# --+ translate topic modeling outcome
lda_30 = gensim.models.wrappers.lda.malletmodel2ldamodel(lda_30)
# --+ term-to-topic probabilities (10 words per topic)
top_terms_line = lda_30.show_topics(num_topics=30, num_words=10)
# --+ rearrange data on top 10 terms per topic
top_terms_m = []
for i in top_terms_line:
    topic_num = i[0]
    prob_terms = i[1].split("+")
    for term_sort, term in enumerate(prob_terms):
        weight = float(term.split("*")[0])
        term = term.split("*")[1].strip('"| ')
        top_terms_m.append([topic_num, term_sort, weight, term])
df = pd.DataFrame(top_terms_m)
# --+ rename columns
old_names = [0, 1, 2, 3]
new_names = ["topic_n", "term_sort", "weight", "term"]
cols = dict(zip(old_names, new_names))
df.rename(columns=cols, inplace=True)
df.set_index(["term_sort", "topic_n"], inplace=True)
df = df.unstack()
# --+ sidewaystable
df_h = pd.DataFrame()
for i in range(30):
    terms = df["term"][i]
    weights = df["weight"][i]
    weights = pd.Series(["( %s )" % j for j in weights])
    df_h = pd.concat([df_h, terms, weights], axis=1)
# --+ write data to file
out_f = os.path.join("analysis", "topicModeling", ".output", "30t_term_topic.tex")
df_h.to_latex(out_f, index=True)
# --+ get transformed corpus as per the lda model
transf_corpus = lda_30.get_document_topics(corpus)
# ----+ rearrange data on document-topic pairs probabilities
doc_topic_m = []
for id, doc in enumerate(transf_corpus):
    for topic in doc:
        topic_n = topic[0]
        topic_prob = topic[1]
        doc_topic_m.append([id, topic_n, topic_prob])  # , topic_prob])
# ----+ get a df
df = pd.DataFrame(doc_topic_m)
# ----+ rename columns
old_names = [0, 1, 2]
new_names = ["doc_id", "topic_n", "prob"]
cols = dict(zip(old_names, new_names))
df.rename(columns=cols, inplace=True)
# ----+ dominant topic
gr = df.groupby("doc_id")
df.loc[:, "max"] = gr["prob"].transform(np.max)
df.loc[:, "first_topic"] = 0
df.loc[df["prob"] == df["max"], "first_topic"] = 1
first_topic = df.loc[df["first_topic"] == 1]
first_topic.set_index("doc_id", inplace=True)
# ----+ arrange data as contingency table
df = df.pivot_table(index="doc_id", columns="topic_n", values="prob", aggfunc=np.mean)
# ----+ write data to files
out_f = os.path.join("analysis", "topicModeling", ".output", "30t_doc_topic_pr.csv")
df.to_csv(out_f, index=True)
out_f = os.path.join("analysis", "topicModeling", ".output", "30t_dominant_topics.csv")
first_topic.to_csv(out_f, index=True)
