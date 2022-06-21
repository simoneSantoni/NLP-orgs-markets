# !/usr/env/bin python3
# -*- encoding utf-8 -*-

"""
------------------------------------------------------------------------------
    _1.py    |    Python script for the discussion of case study #1 on 
             |      the inter-temporal similarity of beer reviews
------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk 

Synopsis : this script presents some key snippets to discuss the first case 
           study of the NLP, orgs, and markets series on the inter-temporal
	       similarity of beer reviews. In terms of scope/structure, the script 
	       carries out the following tasks:

	       - data loading
	       - data wrangling
	       - NLP pipeline
	       - BoW vectorization
	       - post-processing of the TFIDF vectors
           - visualization
           - time series cluster analysis

Notes    : NaN  

"""


# %%
# load libraries
import os
import json
import numpy as np
from scipy.spatial.distance import cosine
import tslearn as tsl
import pandas as pd
import spacy

# %%
# data loading

# path
fdr = "../sampleData/beerReviews/"
# beeer metadata
mt = pd.read_json(os.path.join(fdr, "product_metadata.json"))
# beer reviews
rv = pd.read_json(os.path.join(fdr, "product_reviews.json"))

# %%
# data wrangling

# group reviews within beers
# --+ date as datetime
rv["date"] = pd.to_datetime(rv["date"], errors="coerce")
# --+ resolve NaNs
rv = rv.loc[rv["date"].notnull()]
# --+ sort reviews by date
rv.sort_values(["beer", "date"], inplace=True)
# --+ grouping structure
gr = rv.groupby("beer")
# --+ create a sort field
rv.loc[:, "sort"] = 1
rv.loc[:, "sort"] = gr["sort"].transform(np.cumsum)
# create a review id
rv.loc[:, "review_id"] = np.arange(len(rv))

# %%
# let's pass the reviews through an NLP pipeline

# load a spaCy model
nlp = spacy.load("en_core_web_lg")
# create a dictionary with the available reviews
docs = dict(
    zip(rv.loc[rv["text"].notnull(), "review_id"], rv.loc[rv["text"].notnull(), "text"])
)
# get a list of tokenized docs
docs_tkns = {}
for key, text in docs.items():
    tmp = [
        token
        for token in nlp(text.lower())
        if (not token.is_stop) & (not token.is_punct) & (token.is_alpha)
    ]
    docs_tkns[key] = tmp
    del tmp
# 1-hot encoded BoWs with Pandas DFs
# --+ the empty DF to store the BoWs
oh = pd.DataFrame()
# --+ iterate over tokenized docs
for key, tkns in docs_tkns.items():
    # get the unique tokens
    voc = sorted(set(tkns))
    # one hot encoded BoW
    corpus = pd.DataFrame({pos: 1 for pos in voc}, index=[key])
    # append data
    oh = pd.concat([oh, corpus], axis=0)
    oh.fillna(0, inplace=True)

# %%
# assess inter-temporal similarity of reviews concerning the same beer

# merge one-hot encoded vectors w/beer names
rv.set_index("review_id", inplace=True)
df = pd.merge(rv.loc[:, "beer"], oh, left_index=True, right_index=True, how="inner")
# 
# compare pairs of temporally contiguous reviews
gr = df.groupby("beer")
df