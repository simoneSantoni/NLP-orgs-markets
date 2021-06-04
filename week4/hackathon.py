# %% load libraries
import spacy
import en_core_web_lg

# %% create nlp pipeline
nlp = en_core_web_lg.load()

# %% search for words
nlp.vocab['cr7'].vector

# %%
import gensim.downloader as api

model = api.load("word2vec-google-news-300")  # download the model and return as object ready for use
word_vectors = model.wv #load the vectors from the model