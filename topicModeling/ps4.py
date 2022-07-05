# %%
# import libraries

import numpy as np
import spacy
import tomotopy as tp
import pandas as pd
import matplotlib.pyplot as plt
import gensim
from gensim.models import Phrases


# %%
# initialize NLP pipeline
nlp = spacy.load("en_core_web_lg")

# %%
# load data
df = pd.read_csv("../sampleData/tripadvisorReviews/hotel_reviews.csv")

# %%
# pre-process the data
docs_tokens, tmp_tokens = [], []
for doc in df.loc[:, "Review"].to_list():
    tmp_tokens = [
        token.lemma_
        for token in nlp(doc)
        if not token.is_stop and not token.is_punct and not token.like_num
    ]
    docs_tokens.append(tmp_tokens)
    tmp_tokens = []

# %%
# create a phrases model

common_terms = [
    "of",
    "with",
    "without",
    "and",
    "or",
    "the",
    "a",
    "not",
    "be",
    "to",
    "this",
    "who",
    "in",
    "as",
    "such"
]

bigrams = Phrases(docs_tokens, min_count=5)
trigrams = Phrases(bigrams[docs_tokens], min_count=5)

docs_phrased = [trigrams[bigrams[line] for line in docs_tokens]]

# %%
# create a tomotopy corpus

corpus = tp.utils.Corpus()

for item in docs_tokens:
	    corpus.add_doc(words=item)


# %%
# get the coherence scores for competing models
cvs = {}
for topic_number in range(1, 31, 5):
    mdl = tp.LDAModel(k=topic_number, corpus=corpus, rm_top=10)
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
ax.set_xticks(range(1, 31, 1))
plt.show()


# %% 
# set up the tomotopy model
best_mdl = tp.LDAModel(k=6, corpus=corpus)

# %%
# train the tomotopy model
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
