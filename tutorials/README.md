# Tutorials ― README

## Overview

Below is the set of tutorials students may want to go through to fix some core 
ideas discussed in the online videos/tutorials. The large majority of tutorials
also offer some basic NLP recipes that may be useful for future projects.
(I'll continue to update this as the module progresses; keep an eye on it.)

| File name | Tags                                                     | Data                  | Libraries                   |
| --------- | -------------------------------------------------------- | --------------------- | --------------------------- |
| _0        | NLP pipeline, dictionary creation, corpus transformation | Econ newspaper corpus | spaCy, Gensim               |
| _1        | word embeddings, manipulation of embedding features      | Google News corpus    | spaCy, NumPy/SciPy          |
| _2        | word embeddings, browsing word embeddings                | Google News corpus    | spaCy, Gensim, Scikit-Learn |
| _3        | word embeddings, visualizing embeddings                  | Google News corpus    | spaCy, whatlies             |
| _4        | word embeddings, algebric operations                     | None                  | spaCy                       |

## Learning goals of the individual tutorials

The following sections provide a concise description of each tutorial's learning
goals.
### Tutorial `0` ― textual data preparation and transformation

Learning goals:

1. familiarizing with the functioning of spaCy's NLP pipeline
2. expanding on the default set of stopwords included in spaCy
3. using Gensim to create and store the dictionary and corpus out of a
collection of documents

### Tutorial `1` ― manipulating word vectors with NumPy and SciPy

Learning goals:

1. loading word vectors from trained embeddings
2. browsing word embeddings to sample clusters of words that are semantically
   similar
3. visualizing clusters of words that are semantically similar
4. calculating the distance between any two vectors (that is, their
   semantic (dis-)similarity)

### Tutorial `2` ― browsing word embeddings

Learning goals:

1. loading word vectors from a pre-trained model
2. fetching the word vectors associated with target entities
3. sampling 'alter' words, i.e., words that are in the neighborhood
   of a target vector
4. fetching the word vectors associated with alter words
5. using dimensionality reduction techniques to explore the semantic
   similarity across target and alter words

### Tutorial `3` ― visual exploration of word embeddings' features

Learning goals:

1. charting the semantic relations involving sets of lexical items
2. familiarizing with dedicated libraries for the visualization of word embeddings

### Tutorial `4` ― analogical reasoning with word embeddings

Learning goals:

1. expanding on word embeddings to uncover complex semantic relations involving 
   lexical items

### Tutorial `5` ― using word embeddings as features for ML models

Learning goals:

1. using word embeddings as features in a regression framework
2. picking-up the most suited regression framework to exploit the predictive
   value of word embeddings

__!! To be released after hackathon # 1 !!__