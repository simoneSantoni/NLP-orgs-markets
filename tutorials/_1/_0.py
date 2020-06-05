#! /usr/env/bin python3

"""
Docstring
------------------------------------------------------------------------------
------------------------------------------------------------------------------

Author: Simone Santoni, simone.santoni.1@city.ac.uk

Edits: 
       - created
       - last change

Notes:
"""

# %% load libraries
import os
import stanza 


stanza.download('en') # download English model

nlp = stanza.Pipeline('en') # initialize English neural pipeline

doc = nlp("Barack Obama was born in Hawaii.") # run annotation over a sentence

print(doc)

doc = nlp('This is a test sentence for stanza. This is another sentence.')

for i, sentence in enumerate(doc.sentences):
    print(f'====== Sentence {i+1} tokens =======')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')
