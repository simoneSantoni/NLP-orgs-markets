#!/usr/bin/env python3

"""
Docstring
------------------------------------------------------------------------------
    _0.py    |    Creating a corpus and dictionary out of press data
------------------------------------------------------------------------------

Author: Simone Santoni, simone.santoni.1@city.ac.uk

Edits:
    - created
    - last edit

Notes: the dataset contains 4,384 articles published in the Wall Street
       Journal and the Financial Times that mention one or multiple
       of the following keywords:

       + Computer Software
       + Internet of Things
       + Machine Learning
       + Robotics
       + Technology
       + Computer Science
       + Computers
       + Automation
       + Augmented Reality
       + Big Data
       + Deep Learning
       + Cloud Computing
       + Natural Language Processing
       + Pattern Recognition
       + Analytics
       + Computing

"""

# %% load libraries
# basic operations
import os
import logging
import re
# load data from mongodb
from pymongo import MongoClient
# data analysis/management/manipulation
import numpy as np
import pandas as pd
# nlp pipeline
import spacy
import en_core_web_lg
# building corpus/dictionary
import gensim
from gensim import corpora
from gensim.models import Phrases
from gensim.corpora import Dictionary


# %% check software versions
print("""
spaCy version: {}
Gensim version: {}
""".format(spacy.__version__, gensim.__version__))


# %% work directory
srv = '/home/simone'
prj = 'githubRepos/digital-leadership-center'
fdr = 'transformation'
wd = os.path.join(srv, prj, fdr)
os.chdir(wd)


# %% read data

# create client
'''
in my case, I'm reading data from a Mongo db. If you've got a local copy of
dataset, just use Pandas.

df = pd.read_json(pr_wsj_ft.json)

Note: the 'uri' argument is mandatory to create a pipeline with a Mongo 
server doesn't run on the machine you're sing to run this Python script 
'''
uri = ''
client = MongoClient()

# pick-up db
db = client.digitalTechs

# load the data
df = pd.DataFrame(list(db.press_releases.find()))


# %% clean data

# slice the data
'''
let's focus on the 2009 - 2019 timespan, which concentrates the large majority
of the data.
'''
df = df.loc[df['year'] > 2008]

# basic cleaning
# --+ get timespans
df.loc[:, 'year'] = df['date'].dt.year
# --+ drop column
df.drop(['_id'], axis=1, inplace=True)

# arrange data for sequential lda
# --+ order data by year of publication
df.sort_values('year', inplace=True)
# --+ get stacks by year
data = df.groupby('year').size()
# --+ time slices
time_slices = data.values

# prepare list to pass through spacy
docs = [article.strip().lower() for article in df.text]

# hyphen to underscores
docs = [re.sub(r'\b-\b', '_', text) for text in docs]


# %% NLP pipeline
'''
the pipeline that is displayed below is an example of a simple/standard 
NLP pipeline. spaCy allows the implementation of far more sophisticated 
and rich pipelines -- see this section of the spaCy API concerning 
the pipeline: https://spacy.io/ap
'''
# load spaCy model 'web_lg'
nlp = en_core_web_lg.load()

# expand on spaCy's stopwords
# --+ my stopwrods
my_stopwords = ['\x1c',
                'ft', 'wsj', 'time', 'sec',
                'say', 'says', 'said',
                'mr.', 'mister', 'mr', 'miss', 'ms',
                'inc']
# --+ expand on spacy's stopwords
for stopword in my_stopwords:
    nlp.vocab[stopword].is_stop = True

# tokenize text
docs_tokens, tmp_tokens = [], []

for doc in docs:
    tmp_tokens = [token.lemma_ for token in nlp(doc)
                  if not token.is_stop
                  and not token.is_punct
                  and not token.like_num
                  and not token.like_url
                  and not token.like_email
                  and not token.is_currency
                  and not token.is_oov]
    docs_tokens.append(tmp_tokens)
    tmp_tokens = []

# take into account bi- and tri-grams
'''
Lane and colleagues [1] offer a very effective desccription of what n-grams 
and explaing why we should care about them: 

"An n-gram is a sequence containing up to n elements that have been extracted 
from a sequence of those elements, usually a string. In general the “elements” 
of an n-gram can be characters, syllables, words, or even symbols like “A,” 
“T,” “G,” and “C” used to represent a DNA sequence.6 In this book, we’re only 
interested in n-grams of words, not characters.7 So in this book, when we 
say 2-gram, we mean a pair of words, like “ice cream.” When we say 3-gram, we 
mean a triplet of words like “beyond the pale” or “Johann Sebastian Bach”
r “riddle me this.” n-grams don’t have to mean something special together, 
like com- pound words. They merely have to be frequent enough together to 
catch the attention of your token counters. Why bother with n-grams? As you 
saw earlier, when a sequence of tokens is vectorized into a bag-of-words 
vector, it loses a lot of the meaning inherent in the order of those words. 
By extending your concept of a token to include multiword tokens, n-grams, 
your NLP pipeline can retain much of the meaning inherent in the order of 
words in your statements. For example, the meaning-inverting word “not” will 
remain attached to its neighboring words, where it belongs. Without n-gram 
tokenization, it would be free floating. Its meaning would be associated 
with the entire sentence or document rather than its neighboring words. 
The 2-gram “was not” retains much more of the meaning of the individual 
words “not” and “was” than those 1-grams alone in a bag-of-words vector. A 
bit of the context of a word is retained when you tie it to its neighbor(s) 
in your pipeline."

Concerning the detection of n-grams, Gensim 'model.phrases' 
function (see [2]) allows to detect "common phrases -- aka multi-word 
expressions, word n-gram  collocations -- from a stream of sentences."

[1]: Lane, H., Howard, C., & Hapke, H. M. (2019). Natural Language Processing 
     In action. Manning Publications Co.
     
[2]: https://radimrehurek.com/gensim/models/phrases.html
'''
# --+ get rid of common terms
common_terms = [u'of', u'with', u'without', u'and', u'or', u'the', u'a',
                u'not', 'be', u'to', u'this', u'who', u'in']

# --+ fing phrases as bigrams
bigram = Phrases(docs_tokens,
                 min_count=50,
                 threshold=5,
                 max_vocab_size=50000,
                 common_terms=common_terms)
# --+ fing phrases as trigrams
trigram = Phrases(bigram[docs_tokens],
                  min_count=50,
                  threshold=5,
                  max_vocab_size=50000,
                  common_terms=common_terms)

# uncomment if a tri-grammed, tokenized document is preferred
docs_phrased = [bigram[line] for line in docs_tokens]
#docs_phrased = [trigram[bigram[line]] for line in docs_tokens]

# check outcome of nlp pipeline
print('''
=============================================================================
published article: {}

=============================================================================
tokenized article: {}

=============================================================================
tri-grammed tokenized article: {}

'''.format(docs[1],
           docs_tokens[1],
           docs_phrased[1]))


# %% get corpus & dictionary to use for further nlp analysis

'''
I suggest to prepare the dictionary and the corpus `once for all' -- that is, 
dumping the files that, eventually, will be loaded for further analysis.
'''

# get dictionary and write it to a file
'''
a dictionary is a mapping between words and their integer ids. See Gensim 
documentation here: https://radimrehurek.com/gensim/corpora/dictionary.html
'''
pr_dictionary = Dictionary(docs_phrased)
pr_dictionary.save('/tmp/pr_dictionary.dict')

# get corpus and write it to a file
'''
as per the Gensim documentation, it possible to convert document into the 
bag-of-words (format = list of (token_id, token_count) tuples) via doc2bow
'''
pr_corpus = [pr_dictionary.doc2bow(doc) for doc in docs_phrased]
'''
Gensim offers several utilities to write a corpus of text to a file. 
Personally, I prefer the Matrix Market format [1]

[1]: https://math.nist.gov/MatrixMarket/formats.html
'''
corpora.MmCorpus.serialize('/tmp/pr_corpus.mm', pr_corpus)
