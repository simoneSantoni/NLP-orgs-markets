# !/usr/bin/env python3
# -*- encoding utf-8 -*-
"""
--------------------------------------------------------------------------------
    barebone_doc2vec.py    |    training a doc2vec model with minimal data
--------------------------------------------------------------------------------

author   : simone santoni, simone.santoni.1@city.ac.uk

synopsis : this tutorial pursues the following learning goals:
   
           1. training a doc2vec embedding
	       2. using the doc2vec embedding to appreciate an unseen document

to do    : none

"""

# %%
# libraries
from nltk.tokenize import wordpunct_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# %%
#  Exapmple document (list of sentences)
quotes = [
    "I find television very educating. Every time somebody turns on the set, I go into the other room and read a book",
    "Some people never go crazy. What truly horrible lives they must lead",
    "Be nice to nerds. You may end up working for them. We all could",
    "I do not want people to be very agreeable, as it saves me the trouble of liking them a great deal",
    "I did not attend his funeral, but I sent a nice letter saying I approved of it",
    "So many books, so little time",
]
# %%
# text pre-processing

# tokenization
tkn_quotes = [wordpunct_tokenize(quote.lower()) for quote in quotes]
# document tagging
tgd_quotes = [TaggedDocument(d, [i]) for i, d in enumerate(tkn_quotes)]

# %%
# 

# model creation
model = Doc2Vec(
    tgd_quotes, vector_size=20, window=2, min_count=1, workers=4, epochs=100
)
# let's save the module for future applications
model.save("quote_embedding.model")
# and load it again for the sake of redundancy
model = Doc2Vec.load("quote_embedding.model")

# %%
# project a new document onto the trained space vector
new_quote = "Time is an illusion. Lunchtime doubly so"
new_vector = wordpunct_tokenize(new_quote.lower())
model.docvecs.most_similar(positive=[model.infer_vector(new_vector)], topn=5)

# %%
# that's it
