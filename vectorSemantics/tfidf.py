# !/usr/env/bin python3
# encoding -*- utf-8 -

"""
------------------------------------------------------------------------------
    tfidf.py    |    Some snippets to implement a TFIDF approach
------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk 

Synopsys : The script implements a TFIDF document representation

Notes    : NaN 

"""

# %%
# load libraries
from collections import Counter, OrderedDict
import json
import numpy as np
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')

# %%
# source a corpus of text
in_f = "../sampleData/econNewspaper/ai_in_finance.json"
df = pd.json_normalize([json.loads(line) for line in open(in_f)])

# %%
# get list of the first 100 docs
docs = df.loc[0:99, "text"].tolist()

# %%
# define a vocabulary
# --+ get a list of tokenized docs
docs_tkns = []
for doc in docs:
    tmp = [
        token
        for token in nlp(doc)
        if (not token.is_stop) & (not token.is_punct) & (token.is_alpha)
    ]
    docs_tkns.append(tmp)
    del tmp
# --+ build the vocabulary and display it
voc = sorted(set(sum(docs_tkns, [])))
print(voc)
# let's project the individual docs onto the vocabulary
# --+ an empty container for the TFIDF transformations
vector_space = []
# --+ let's store the vectors associated to the individual docs
for doc in docs_tkns:
    vector = OrderedDict((token, 0) for token in voc)
    tkns_count = Counter(doc)
    for k, v in tkns_count.items():
        tf = np.log10(v + 1)
        docs_with_key = 0
        for doc_ in docs_tkns:
            if k in doc_:
                docs_with_key += 1
        if docs_with_key:
            idf = np.log10(len(docs_tkns) / docs_with_key)
        else:
            idf = 0
        vector[k] = np.round(tf * idf, 4)
    vector_space.append(vector)

