#!/usr/env/bin python3
# -*- encoding utf-8 -*-
"""
------------------------------------------------------------------------------
    _8.py    |    implementation of the semantic axis method
------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk

Synopsis : The script shows how to implement LDA Topic Modeling with Tomotopy

To do    : None

"""

# %% load libraries
import os
import pandas as pd
import spacy
import en_core_web_lg
import tomotopy as tp

# %% initialize a spaCy's pipeline
nlp = en_core_web_lg.load()

# %% load General Inquirer categories of words
df = pd.read_csv('./inquireaugmented.csv')

# %% custom function that creates a semantic axis


# %% get the affect score for a sample of unseen words


