# Tutorials ― README

## Overview

Below is the set of tutorials students may want to go through to fix some core 
ideas discussed in the online videos/tutorials. The large majority of tutorials
also offer some basic NLP recipes that may be useful for future projects.
(I'll continue to update this as the module progresses; keep an eye on it.)

| File name | Tags                                                     | Data                     | Libraries                   |
| --------- | -------------------------------------------------------- | ------------------------ | --------------------------- |
| _0        | NLP pipeline, dictionary creation, corpus transformation | Econ newspaper corpus    | spaCy, Gensim               |
| _1        | word embeddings, manipulation of embedding features      | Google News corpus       | spaCy, NumPy/SciPy          |
| _2        | word embeddings, browsing word embeddings                | Google News corpus       | spaCy, Gensim, Scikit-Learn |
| _3        | word embeddings, visualizing embeddings                  | Google News corpus       | spaCy, whatlies             |
| _4        | word embeddings, algebric operations                     | None                     | spaCy                       |
| _5        | minimum edit distance, string comparison                 | None                     | jellyfish                   |
| _6        | topic modeling                                           | Econ newspaper corpus    | spaCy, Gensim, Mallet       |
| _7        | topic modeling                                           | Tripadvisor hotel review | spaCy, tomotopy             |

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

### Tutorial `5` ― string comparison via minimum edit distance

Learning goals:

1. appreciate the approximate or phonetic distance between
two string is a common task when it comes to manipulate strings
2. familiarizing with the jellyfish
library for Python, which implements several distance metrics.

One of the metrics available in the jellyfish library is the Levenshtein
Distance / Minimum Edit Distance, the number of insertions, deletions, and
substitutions required to change one word to another.

One possible application is identifying successful patch/code submissions on the
part of developers or data scientists working in a team setting (e.g., GitHub
teams). For example, the members of the Linux Kernel community exchange their
code using the mailing list of the project, while very few members have the
power to accept (i.e., to commit) a patch/code submissions. This raises concerns
on who contributed to what and to what extent.

This script applies the Levensthein Distance/Minimum Edit Distance to appreciate
the similarity between some code included in a fictional email and the commit
that is visible in a fictional, reference repository.

### Tutorial `6` – topic modeling with Gensim and Mallet

Learning goals:

1. modeling a corpus of economic newspaper articles using Gensim & Mallet’s
implementation of the Latent Dirichlet Allocation, the most popular algorithm in
the area of Topic Modeling
2. exploring and evaluating competing Topic Models (i.e., models retaining different 
number of topics)

The Python script included in the tutorial contains the code necessary to
reproduce the analyses reported in the book chapter [Lanzolla, G., Santoni, S.,
and Tucci, C.  “Unlocking value from AI in financial services: strategic and
organizational tradeoffs vs. media narratives”. 2021](https://github.com/simoneSantoni/applied-NLP-smm694/blob/60ff0a0857e585f42f200fc1fb962911939be4da/data/econNewspaper/lanzolla_santoni_tucci.pdf)

### Tutorial `7` – topic modeling with Tomotopy

Learning goals:

1. Estimating an LDA model with Tomotopy
2. Integrating a spaCy pipeline with Tomotopy estimation capabilities

This tutorial uses a sample of 20+ reviews from Tripadvisor.