#!/usr/env/bin python3
#-*-encoding utf-8 -*-
"""
------------------------------------------------------------------------------
    wb_2_I.py    |    Python script for session 2, WordNet with NLTK 
------------------------------------------------------------------------------

Author     : Simone Santoni, simone.santoni.1@city.ac.uk 

Credits to : Bird, Klein & Loper (2009). Natural Language Processing with
             Python

Synopsis   : the script covers the following points:

             A. discovering synsets
             B. navigating the abstract-2-concrete continuum  
             C. examples of lexical relations

TODO       : None

"""

# %% laod libraries 
import nltk
from nltk.corpus import wordnet as wn

# %% - A - discovering synsets
# get the synsets for a focal word
wn.synset('motorcar.n.01')
# get the lemmas included in a certain synset
wn.synset('car.n.01').lemma_names()
# get the definition for the synset 
wn.synset('car.n.01').definition()
# example referring to the sysnet
wn.synset('car.n.01').examples()
"""
Although definitions help humans to understand the intended meaning of a synset,
the words of the synset are often more useful for our programs. To eliminate
ambiguity, we will identify these words as car.n.01.automobile,
car.n.01.motorcar, and so on. This pairing of a synset with a word is called a
lemma. We can get all the lemmas for a given synset , look up a particular lemma
, get the synset corresponding to a lemma , and get the “name” of a lemma
"""
# get the lemmas for synset 'car.n.01'
wn.synset('car.n.01').lemmas()
# get the lemma for an item in the synset of interest
wn.lemma('car.n.01.automobile').synset()
# get the word associated with the lemma for the item of interest
wn.lemma('car.n.01.automobile').name
"""
Unlike the words automobile and motorcar, which are unambiguous and have one
syn- set, the word car is ambiguous, having five synsets
"""
# synsets to which a focal word is associated with
wn.synsets('car')
# get the lemmas'name for each sysnset
for synset in wn.synsets('car'):
    print(synset.lemma_names)

# %% - B - navigating the abstract-2-concrete continuum  
"""
WordNet makes it easy to navigate between concepts. For example, given a concept
like motorcar, we can look at the concepts that are more specific—the
(immediate) hyponyms.
"""
# given a synset
motorcar = wn.synset('car.n.01')
# get all types of motorcars
types_of_motorcar = motorcar.hyponyms()
# slice types
types_of_motorcar[26]
# print all words in the 'lower-level' synset
sorted([lemma.name() for synset in types_of_motorcar for lemma in synset.lemmas()])
"""
We can also navigate up the hierarchy by visiting hypernyms. Some words have
multiple paths, because they can be classified in more than one way.
"""
# get more 'abstract' concepts
motorcar.hypernyms()
# get the pathways
paths = motorcar.hypernym_paths()
# evaluate the number of pathways
len(paths)
# get the synsets nested in each path
[synset.name for synset in paths[0]]
[synset.name for synset in paths[1]]

# %% - C - examples of lexical relations
"""
Hypernyms and hyponyms are called lexical relations because they relate one
synset to another. These two relations navigate up and down the “is-a”
hierarchy. Another important way to navigate the WordNet network is from items
to their components (meronyms) or to the things they are contained in
(holonyms). For example, the parts of a tree are its trunk, crown, and so on;
these are the part_meronyms().
"""
wn.synset('tree.n.01').part_meronyms()
"""
The substance a tree is made of includes heartwood and sapwood, i.e., the
substance_meronyms().
"""
wn.synset('tree.n.01').substance_meronyms()
"""
A collection of trees forms a forest, i.e., the
member_holonyms()
"""
wn.synset('tree.n.01').member_holonyms()
"""
There are also relationships between verbs. For example, the act of walking involves the act of stepping, so walking entails stepping. Some verbs have multiple entailments:
"""
wn.synset('walk.v.01').entailments()
wn.synset('eat.v.01').entailments()
wn.synset('tease.v.03').entailments()
"""
Some lexical relationships hold between lemmas, e.g., antonymy:
"""
wn.lemma('supply.n.02.supply').antonyms()
wn.lemma('rush.v.01.rush').antonyms() 
wn.lemma('horizontal.a.01.horizontal').antonyms()
wn.lemma('staccato.r.01.staccato').antonyms()
"""
We have seen that synsets are linked by a complex network of lexical relations.
Given a particular synset, we can traverse the WordNet network to find synsets
with related meanings. Knowing which words are semantically related is useful
for indexing a col- lection of texts, so that a search for a general term such
as vehicle will match documents containing specific terms such as limousine.
"""