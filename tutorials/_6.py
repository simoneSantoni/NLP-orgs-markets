#!/usr/bin/env python3
# -*-encoding utf-8 -*-
"""
------------------------------------------------------------------------------
    _6.py    |    LDA on an economic newspaper corpus
------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk

Synopsis : The corpus oftext contains documents published over circa 10 years.
           This LDA model analyzes the documents as a pooled cross section
           model (econometrics guys will appreciate that).

To do    : None

"""

# %% load libraries
import os
import pickle
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import gensim
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamodel import LdaModel, CoherenceModel
from gensim.models.wrappers import LdaMallet, ldamallet
from gensim.similarities import MatrixSimilarity

# %% working dir
os.chdir("../data/econNewspaper")

# %% viz options
plt.style.use("seaborn-bright")
rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})
rc("text", usetex=True)

# %% external software
mallet_path = "/home/simone/.mallet/mallet-2.0.8/bin/mallet"

# %% load data

# collection in Mongo
# --+ open pipeline
client = MongoClient()
# --+ pick-up db
db = client.digitalTechs
# --+ load the data
df = pd.DataFrame(list(db.press_releases.find()))

# load an existing dictionary for the corpus
in_f = os.path.join("pr_dictionary.dict")
dictionary = Dictionary.load(in_f)

# load an existing representation for the corpus
in_f = os.path.join("pr_corpus.mm")
corpus = MmCorpus(in_f)

# load an existing transformation for the corpus that contains n-grams
in_f = os.path.join("pr_docs_phrased.pickle")
with open(in_f, "rb") as pipe:
    docs_phrased = pickle.load(pipe)

# %% clean data read from Mongo

# basic cleaning
# --+ get timespans
df.loc[:, "year"] = df["date"].dt.year
# --+ slice the data
"""
let's focus on the 2013 - 2019 timespan, which concentrates the large majority
of the data.
"""
df = df.loc[df["year"] >= 2009]
# --+ drop column
df.drop(["_id"], axis=1, inplace=True)

# %% exploratory data analysis

# barchart of the distribution of articles over time
# --+ data series
x = np.arange(2013, 2020, 1)
y0 = df.loc[(df["outlet"] == "ft") & (df["year"] >= 2013)].groupby("year").size().values
y0[-1] = 336
y1 = (
    df.loc[(df["outlet"] == "wsj") & (df["year"] >= 2013)].groupby("year").size().values
)
# --+ labels
x_labels = ["%s" % i for i in x]
y_labels = ["%s" % i for i in np.arange(0, 1400, 200)]
for i, s in enumerate(y_labels):
    if len(s) > 3:
        y_labels[i] = "{},{}".format(s[0], s[1:])
    else:
        pass
# --+ create figure
fig = plt.figure(figsize=(6, 4))
# --+ populate figure with a plot
ax = fig.add_subplot(1, 1, 1)
# --+ plot data
ax.bar(
    x, y0, color="k", width=0.6, alpha=0.50, edgecolor="white", label="Financial Times"
)
ax.bar(
    x,
    y1,
    color="k",
    width=0.6,
    alpha=0.25,
    edgecolor="white",
    bottom=y0,
    label="Wall Street Journal",
)
# --+ axis properties
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=14, rotation="vertical")
ax.set_xlabel("Year", fontsize=14)
ax.set_yticklabels(y_labels, fontsize=14)
ax.set_ylabel("Counts of documents", fontsize=14)
# --+ hide all spines while preserving ticks
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")
# --+ grid
ax.grid(True, ls="--", axis="y", color="white")
# --+ legend
plt.legend(loc="best")
# --+ save plot
out_f = os.path.join(
    "scripts", "analysis", "topicModeling", ".output", "articles_over_time.pdf"
)
plt.savefig(out_f, bbox_inches="tight", pad_inches=0)


# %% topic modeling ― explore model validity

# define function to explore a gamut of competing models
def compute_coherence_values(
    _dictionary, _corpus, _texts, _limit, _start, _step, _path, _seed
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
        model = gensim.models.wrappers.Lda(
            mallet_path,
            corpus=_corpus,
            num_topics=num_topics,
            id2word=_dictionary,
            random_seed=_seed,
        )
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model, texts=_texts, dictionary=_dictionary, coherence="c_v"
        )
        coherence_values.append(coherencemodel.get_coherence())
    # return output
    return model_list, coherence_values


# collect coherence scores as the number of retained topics change
"""
I make two searches:

    - 5 - 10 topics, step = 1
    - 10 - 30 topics, step = 5

"""

# 5 - 9 topic models
# --+ search grid
limit, start, step = 10, 5, 1

# --+ run function
ml_5_9, cv_5_9 = compute_coherence_values(
    _dictionary=dictionary,
    _corpus=corpus,
    _texts=docs_phrased,
    _start=start,
    _limit=limit,
    _step=step,
    _seed=123,
    _path=mallet_path,
)

# 10 - 30 topic models
# --+ search grid
limit, start, step = 35, 10, 5
# --+ run function
ml_10_30, cv_10_30 = compute_coherence_values(
    _dictionary=dictionary,
    _corpus=corpus,
    _texts=docs_phrased,
    _start=start,
    _limit=limit,
    _step=step,
    _seed=123,
    _path=mallet_path,
)

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


# %% topic model estimation
"""
I focus on two models:
    - 8 topics, ~ local optimum
    - 30 topic, ~ global optimum
"""

# model with 8 topics
# --+ estimate model
lda_8 = LdaMallet(
    mallet_path, corpus=corpus, id2word=dictionary, num_topics=8, random_seed=123
)
# --+ print topics (20 words per topic)
lda_8.print_topics(num_topics=8, num_words=20)
# --+ translate topic modeling outcome
lda_8 = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_8)

# --+ term-to-topic probabilities (10 words per topic)
top_terms_line = lda_8.show_topics(num_topics=8, num_words=10)
# ----+ rearrange data on top 10 terms per topic
top_terms_m = []
for i in top_terms_line:
    topic_num = i[0]
    prob_terms = i[1].split("+")
    for term_sort, term in enumerate(prob_terms):
        weight = float(term.split("*")[0])
        term = term.split("*")[1].strip('"| ')
        top_terms_m.append([topic_num, term_sort, weight, term])
df = pd.DataFrame(top_terms_m)
# ----+ rename columns
old_names = [0, 1, 2, 3]
new_names = ["topic_n", "term_sort", "weight", "term"]
cols = dict(zip(old_names, new_names))
df.rename(columns=cols, inplace=True)
df.set_index(["term_sort", "topic_n"], inplace=True)
df = df.unstack()
# ----+ sidewaystable
df_h = pd.DataFrame()
for i in range(8):
    terms = df["term"][i]
    weights = df["weight"][i]
    weights = pd.Series(["( %s )" % j for j in weights])
    df_h = pd.concat([df_h, terms, weights], axis=1)
# ----+ write data to file
out_f = os.path.join(
    "scripts", "analysis", "topicModeling", ".output", "8t_term_topic.tex"
)
df_h.to_latex(out_f, index=True)
# --+ get transformed corpus as per the lda model
transf_corpus = lda_8.get_document_topics(corpus)
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
out_f = os.path.join(
    "scripts", "analysis", "topicModeling", ".output", "8t_doc_topic_pr.csv"
)
df.to_csv(out_f, index=True)
out_f = os.path.join(
    "scripts", "analysis", "topicModeling", ".output", "8t_dominant_topics.csv"
)
first_topic.to_csv(out_f, index=True)


# model with 30 topics
# ----+ estimate model
_path = "/home/simone/.mallet/mallet-2.0.8/bin/mallet"
lda_30 = Lda(
    mallet_path, corpus=corpus, id2word=dictionary, num_topics=30, random_seed=123
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
