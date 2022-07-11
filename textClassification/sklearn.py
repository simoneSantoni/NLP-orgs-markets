# /usr/env/bin python3
# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------
    sklearn.py  |  Text Classification with Scikit-Learn and doc-to-topic
                |  probabilities
------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk 

Synopsis : this script uses scikit-learn to train a text classifier fed with 
           document-to-topic probabilities from an LDA model. The dataset 
           contains 20,491 rating-review pairs from Tripadvisor.

Notes    : none

"""

# %%
# load libraries
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
import tomotopy as tp
import spacy

# %%
# load the dataset
df = pd.read_csv("../sampleData/tripadvisorReviews/hotel_reviews.csv")

# %%
# explore the dataset

# theme
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
# distribution of items across ratings
sns.catplot(x="Rating", data=df, kind="count")
# distribution of review length across ratings
sns.violinplot(x="Rating", y=np.log(df.loc[:, "Review"].str.len()), data=df)

# %%
# data transformation

# review class
df.loc[:, "label"] = np.nan
# good review
df.loc[df["Rating"] < 3, "label"] = int(0)
df.loc[df["Rating"] > 3, "label"] = int(1)
# distribution of reviwes across labels
sns.catplot(x="label", data=df.loc[df["label"].notnull()], kind="count")

# %%
# data sampling

# bad reviews
bad = df.loc[df["label"] == 0, ["Review", "label"]]
# good reviews
good = df.loc[df["label"] == 1, ["Review", "label"]]
good = good.sample(n=len(bad), random_state=42)
# combine good and bad reviews
s = pd.concat([bad, good])

# %%
# pre-process the data

# nlp pipeline
nlp = spacy.load("en_core_web_lg")
# pass reviews through the pipeline
docs = nlp.pipe(
    s.loc[:, "Review"].str.lower(), n_process=2, batch_size=500, disable=["tok2vec"],
)
tkns_docs = []
for doc in docs:
    tmp = []
    for token in doc:
        if (
            token.is_stop == False
            and token.is_punct == False
            and token.like_num == False
        ):
            tmp.append(token.lemma_)
    tkns_docs.append(tmp)
    del tmp


# %%
# train an LDA model

# create a corpus using tp utilities
corpus = tp.utils.Corpus()
# populate the corpus
for item in tkns_docs:
    corpus.add_doc(words=item)
# search for the best fitting model
mf = {}
for i in range(10, 260, 100):
    print("Working on the model with {} topics ...\n".format(i), flush=True)
    mdl = tp.LDAModel(k=i, corpus=corpus, min_df=5, rm_top=5, seed=42)
    mdl.train(0)
    for j in range(0, 10, 10):
        mdl.train(10)
        print("Iteration: {}\tLog-likelihood: {}".format(j, mdl.ll_per_word))
    coh = tp.coherence.Coherence(mdl, coherence="u_mass")
    mf[i] = coh.get_score()
    mdl.save("k_{}".format(i), True)

    # %%
