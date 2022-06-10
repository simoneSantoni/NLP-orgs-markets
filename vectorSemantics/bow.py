# !/usr/env/bin python3
# encoding -*- utf-8 -

"""
------------------------------------------------------------------------------
    bow.py    |    Some snippets for the BoW approach
------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk 

Synopsys : The script implements a barebone BoW and a couple of slighlty more 
           advanced BoW approaches.

Notes    : NaN 

"""

# %%
# load libraries
from collections import Counter, OrderedDict, defaultdict
from typing import Dict, List, Tuple
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
import spacy
import pandas as pd

# %%
# a bare bone BoW implementation --- single document case

# sample text
text = """
BlackRock, Inc. is an American multinational investment management corporation 
based in New York City. Founded in 1988, initially as a risk management and 
fixed income institutional asset manager, BlackRock is the world's largest 
asset manager, with US$10 trillion in assets under management as of January 
2022. BlackRock operates globally with 70 offices in 30 countries and clients 
in 100 countries"
"""
# text tokenizations
tkns = TreebankWordTokenizer().tokenize(text)
# let's transform the sample text into a BoW
bow = Counter(tkns)
print(bow)
# let's produce some term frequencies to facilitate cross-document comparison
tf = {k: np.round(v / len(tkns), 4) for k, v in bow.items()}


# %%
# a bare bone BoW implementation --- multiple documents case

# sample docs
text_0 = """
The company was started by Adolf Dassler in his mother's house; he was joined 
by his elder brother Rudolf in 1924 under the name Gebrüder Dassler 
Schuhfabrik ("Dassler Brothers Shoe Factory"). Dassler assisted in the 
development of spiked running shoes (spikes) for multiple athletic events. To 
enhance the quality of spiked athletic footwear, he transitioned from a 
previous model of heavy metal spikes to utilising canvas and rubber. Dassler 
persuaded U.S. sprinter Jesse Owens to use his handmade spikes at the 1936 
Summer Olympics. In 1949, following a breakdown in the relationship between 
the brothers, Adolf created Adidas, and Rudolf established Puma, which became 
Adidas' business rival
"""
text_1 = """
The company was founded on January 25, 1964, as "Blue Ribbon Sports", by Bill 
Bowerman and Phil Knight, and officially became Nike, Inc. on May 30, 1971. 
The company takes its name from Nike, the Greek goddess of victory. Nike 
markets its products under its own brand, as well as Nike Golf, Nike Pro, 
Nike+, Air Jordan, Nike Blazers, Air Force 1, Nike Dunk, Air Max, Foamposite, 
Nike Skateboarding, Nike CR7, and subsidiaries including Jordan Brand and 
Converse. Nike also owned Bauer Hockey from 1995 to 2008, and previously owned 
Cole Haan, Umbro, and Hurley International. In addition to manufacturing 
sportswear and equipment, the company operates retail stores under the 
Niketown name. Nike sponsors many high-profile athletes and sports teams 
around the world, with the highly recognized trademarks of "Just Do It" and 
the Swoosh logo.
"""
text_2 = """
Puma SE, branded as Puma, is a German multinational corporation that designs 
and manufactures athletic and casual footwear, apparel and accessories, which 
is headquartered in Herzogenaurach, Bavaria, Germany. Puma is the third largest 
sportswear manufacturer in the world. The company was founded in 1948 by 
Rudolf Dassler. In 1924, Rudolf and his brother Adolf "Adi" Dassler had 
jointly formed the company Gebrüder Dassler Schuhfabrik (Dassler Brothers Shoe 
Factory). The relationship between the two brothers deteriorated until the two 
agreed to split in 1948, forming two separate entities, Adidas and Puma. Both 
companies are currently based in Herzogenaurach, Germany.
"""
docs = [text_0, text_1, text_2]
# define a vocabulary
# --+ get a list of tokenized docs
docs_tkns = [sorted(TreebankWordTokenizer().tokenize(doc)) for doc in docs]
# --+ build the vocabulary and display it
voc = sorted(set(sum(docs_tkns, [])))
print(voc)
# let's project the individual docs onto the vocabulary
# --+ an empty container for the BoW transformations
vector_space = []
# --+ let's store the vectors associated to the individual docs
for doc in docs_tkns:
    vector = OrderedDict((token, 0) for token in voc)
    tkns_count = Counter(doc)
    for k, v in tkns_count.items():
        vector[k] = v
    vector_space.append(vector)
    del vector

# %%
# a BoW implementation drawing on a spaCy pipeline

# load a model of the language
nlp = spacy.load("en_core_web_sm")
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
# --+ an empty container for the BoW transformations
vector_space = []
# --+ let's store the vectors associated to the individual docs and display it
for doc in docs_tkns:
    vector = OrderedDict((token, 0) for token in voc)
    tkns_count = Counter(doc)
    for k, v in tkns_count.items():
        vector[k] = v
    vector_space.append(vector)
print(vector_space)

# %%
# a BoW implementation that expands the vocabulary as new docs are processed

# here's our function
def doc2bow(tkns_: List[str], voc_: Dict[str, int]) -> List[Tuple[int, int]]:
    """_summary_

	Args:
	    tkns_ (_type_): _description_
	    voc_ (_type_): _description_

	Returns:
	    _type_: _description_
	"""
    tkns_count = defaultdict(int)
    for tkn in tkns_:
        if tkn not in voc_:
            voc_[tkn] = len(voc_)
        tkns_count[voc_[tkn]] += 1

    return list(tkns_count.items())


# let's deploy the function
# --+ an empty container for the BoW transformations
voc = {}
# --+ BoW representation for the first doc
print(doc2bow(TreebankWordTokenizer().tokenize(docs[0]), voc))
# --+ BoW representation for the second doc
print(doc2bow(TreebankWordTokenizer().tokenize(docs[1]), voc))


# %%
# 1-hot encoded BoWs with Pandas DFs

# the empty DF to store the BoWs
oh = pd.DataFrame()

for i, doc in enumerate(docs):
    # tokenize the doc
    tkns = TreebankWordTokenizer().tokenize(doc)
    # get the unique tokens
    voc = sorted(set(tkns))
    # one hot encoded BoW
    corpus = pd.DataFrame({k: 1 for k in voc}, index=[i])
    # append data 
    oh = pd.concat([oh, corpus], axis=0)
    oh.fillna(0, inplace=True)
