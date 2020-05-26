# !/usr/env/bin python3

"""
Docstring
-------------------------------------------------------------------------------
    wb_1.py    |    webinar #1, Python script
-------------------------------------------------------------------------------

Author: Simone Santoni, simone.santoni.1@city.ac.uk

Edits: create ; last edit .

Contents:

    - regex
    - words, text corpus, text corpora
    - text normalization
    - minimum edit distance

"""

# %% load libraries
import os
import re

# %% regex
'''

'''

# -- sample sentences
s0 = 'Apple laptops are for starbucksers'

s1 = 'Did an apple really fall on Isaac Newton\'s head?'

s2 = 'How do I fix my Apple MacBook Pro? I spilled apple juice'\
     'over the keyboard'

s3 = 'The brand-new MacBookPro13-in costs £ 1,500'

s4 = 'Each glass of apple juice takes 3 apples'

s5 = 'run ran run'

s6 = 'The CEO of alfa announced the acquisition of beta.'\
     'The reaction of stakeholders has been positive'

# -- regex are case sensitive 

q = re.search('apple', s0)
print(q)

q = re.search('apple', s1)
print(q.group(0))

q = re.search('[A-Z]', s0)
print(q)

q = re.search('[A-Z]', s1)
print(q)

q = re.search('apple', s2)
print(q)


# -- using the ^ to negate a pattern
q = re.search('[^A-Z]', s0)
print(q)

q = re.search('[^Ap]', s0)
print(q)

q = re.search('[^a-z]', s0)
print(q)

q = re.search('[^a-z, ^A-Z]', s0)
print(q)

q = re.search('[^a-z, ^A-Z]', s3)
print(q)

# -- the preceding character or nothing'

q = re.search('apples?', s4)
print(q)


# -- between 'a' and 'c'
q = re.search("r.n", s5)
print(q)


# -- pattern repetition
q = re.search('[0-9]', s3)
print(q)

q = re.search('[0-9][0-9]*', s3)
print(q)


# -- one or more occurrences of the immediately preceding
#    character or regular expression

q = re.search('[0-9]+', s3)
print(q)


# -- using the `^` symbol to match the start of a line
q = re.search('^The', s6)
print(q)


# -- disjunctions
q = re.search('alfa|beta', s6)
print(q)


# -- grouping
q = re.search('r(un|an)', s5)
print(q)



# %% words, text corpus, and text corpora

# -- a corpus of text as a string
print(s0)

# -- a text corpora
tc = [s0, s1, s2]


# %% text normalization

# -- tokenizing
tokens = s0.split(' ')

# -- normalizing word formats
base_form_of_verbs = s5.replace('ran', 'run')

unique_entities = s4.replace('apples', 'apple')

# -- segmenting sentences
segmnted_corpus = [word.strip() for word in s6.split('.')]

