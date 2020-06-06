#! /usr/env/bin python3

"""
Docstring
------------------------------------------------------------------------------
    _0.py    |    example of Stanza nlp pipeline
------------------------------------------------------------------------------

Author: Simone Santoni, simone.santoni.1@city.ac.uk

Notes: Mainly, there are four arguments behind the choice to use Stanza
       instead of competing libraries such as spaCy:

       1. Stanza builds upon PyTorch. Implication: present a GPU-enabled
          machine, Stanza's performance increases

       2. Stanza `neural` pipeline is more accurate than spaCy's one [1] 
          (but slower). In the spaCy universe, there's a library that allows
          to use Stanza as a spaCy pipeline [2]

       3. availability of multi-word tokens, which allow for more accurate
          and nuanced representation of the linkages connecting a token and its
          underlying syntactic words (or vice versa)

       4. seamless integration with Stanford CoreNLP [1], a very established,
          mature project offering NLP capabilities and state of the art
          text analytics (APIs available for several languages; CoreNLP can
          also run as a web service)

      [1]: https://stanfordnlp.github.io/CoreNLP/
      
      [2]: https://spacy.io/universe/project/spacy-stanza
"""

# %% load libraries
import re
from pprint import pprint as pp
import pandas as pd
import stanza 


# %% read review data covering airlines

# repo
url = 'https://raw.githubusercontent.com/quankiquanki/'\
      'skytrax-reviews-dataset/master/data/airline.csv'

# get a df
df = pd.read_csv(url)

'''
The column containing review data is 'content'

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 41396 entries, 0 to 41395
Data columns (total 20 columns):
 #   Column                         Non-Null Count  Dtype  
---  ------                         --------------  -----  
 0   airline_name                   41396 non-null  object 
 1   link                           41396 non-null  object 
 2   title                          41396 non-null  object 
 3   author                         41396 non-null  object 
 4   acuthor_country                 39805 non-null  object 
 5   date                           41396 non-null  object 
 6   content                        41396 non-null  object 
 7   aircraft                       1278 non-null   object 
 8   type_traveller                 2378 non-null   object 
 9   cabin_flown                    38520 non-null  object 
 10  route                          2341 non-null   object 
 11  overall_rating                 36861 non-null  float64
 12  seat_comfort_rating            33706 non-null  float64
 13  cabin_staff_rating             33708 non-null  float64
 14  food_beverages_rating          33264 non-null  float64
 15  inflight_entertainment_rating  31114 non-null  float64
 16  ground_service_rating          2203 non-null   float64
 17  wifi_connectivity_rating       565 non-null    float64
 18  value_money_rating             39723 non-null  float64
 19  recommended                    41396 non-null  int64  
dtypes: float64(8), int64(1), object(11)
memory usage: 6.3+ MB
'''

# get a list to pass through the Stanza pipeline
# --+ custom 'cleaning' function
def cleaning(_string):
    '''
    : argument     : string 's'
    : return clean : clean version of 's' (lower case, no non-alpha characters)
    '''
    # purge non alpha characters
    alpha = re.sub("[^A-Za-z']+", ' ', str(_string))
    return alpha.lower()


# --+ get a list
docs = [cleaning(item) for item in df.content.values]


# ----+ example of review
pp("""
===============================================================================
Raw text
--------------------------------------------------------------------------------
{}
===============================================================================
Clean text
--------------------------------------------------------------------------------
{}""".format(df.content[99], docs[99]))


# %% load english model of the language

# download English model (once for all)
#stanza.download('en') 

# initialize English neural pipeline
'''
One of the most central options to pass to stanaza.Pipelin() is 'use_gpu'. If
'True', Stanza attempts to use a GPU if available. Set this to 'False' if you
are in a GPU-enabled environment but want to explicitly keep Stanza from using
the GPU.

The individual components of the pipeline can be optionally enacted using the
'processors' option (similarly to spaCy's 'disable' option)
'''

nlp = stanza.Pipeline('en')

'''
The log ― see below displayed example ― informs you that I'm running Stanza on
the cpu (alternatively, you could 'gpu') and all processores have beeen
loaded. 

2020-06-06 09:44:38 INFO: Loading these models for language: en (English):
=========================
| Processor | Package   |
-------------------------
| tokenize  | ewt       |
| pos       | ewt       |
| lemma     | ewt       |
| depparse  | ewt       |
| ner       | ontonotes |
=========================

2020-06-06 09:44:38 INFO: Use device: cpu
2020-06-06 09:44:38 INFO: Loading: tokenize
2020-06-06 09:44:38 INFO: Loading: pos
2020-06-06 09:44:39 INFO: Loading: lemma
2020-06-06 09:44:39 INFO: Loading: depparse
2020-06-06 09:44:40 INFO: Loading: ner
2020-06-06 09:44:42 INFO: Done loading processors!
'''

# %% Let's see the individual processors of Stanza's neural pipeline

'''
We see some of the components of Stanza in the following order:

a. Tokenizer

b. MultiWordTokens

c. Part-of-speech tagger

d. Lemmatizer


Further components, namely, Deeparser and NER, are available in Stanza. We'll
see them in week 6.
'''

# a. Tokenizer

'''
appealing feature of the library: Stanza performs tokenization and sentence 
segmentation at the same time
'''

for doc in docs[0:2]:
    for i, sentence in enumerate(nlp(doc).sentences):
        print(f'___________ Sentence {i+1} tokens ____________')
        print(*[f'id: {token.id}\ttext: {token.text}' 
                for token in sentence.tokens], sep='\n')


tokens = []

for i, doc in enumerate(docs[0:10]):
    for j, sentence in enumerate(nlp(doc).sentences):
        tmp_tokens = [t.text for t in sentence.tokens]
        tokens.append([i, [j, tmp_tokens]])


# b. MultiWordTokens

'''
In Latin languages, there are phonological/ortographic words that result
from the combination of multiple syntactic words. For example: French 
phonological word 'au' is associated with two syntactic words, namely,
'a'' and 'le'. The nlp pipeline should get rid of these multi-word-tokens
because text annotation is built to work on individual tokens. 
'''

# --+ load the 'French' model
stanza.download('fr')
fr_nlp = stanza.Pipeline('fr')

# --+ sentence
text = 'je vais au supermarché'

# --+ process the sentence
doc = fr_nlp(text)

# --+ print word-token pairs
'''
the output of 'fr_nlp' reports two words associated with the token 'au'
'''
for word in doc.sentences[0].words:
    print(f'word: {word.text.ljust(15)} parent token: {word.parent.text}')


# c. Part-of-speech tagger

'''
getting lemmas is straightforward. Let's consider the piece of French text
first.
'''

for word in doc.sentences[0].words:
    print(f'''word: {word.text}
                  upos: {word.upos}
                  xpos: {word.xpos}
                  feats: {word.feats if word.feats else "_"}''')


# d. Lemmatizer

'''
getting lemmas is straightforward. Let's consider the piece of French text
first.
'''

for word in doc.sentences[0].words:
    print(f'word: {word.text.ljust(15)} lemma: {word.lemma}')


'''
let's suppose we want to lemmatize the corpus of reviews covering some 
airlines
'''

lemmas = []

for i, doc in enumerate(docs[0:10]):
    for j, sentence in enumerate(nlp(doc).sentences):
        tmp_lemma = [word.lemma for word in sentence.words]
        lemmas.append([i, [j, tmp_lemma]])

