# !/usr/env/bin python3
# encoding -*- utf-8 -

"""
------------------------------------------------------------------------------
    spacy_nlp_pipeline.py    |    A typical text pre-processing pipeline
------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk 

Synopsys : The script implments a barebone tokenizer in pure Python. Sample 
           data from IMDB are used to illustrate the functioning of the 
	   tokenizer.

Notes    : NaN 

"""

# %% Import libraries
import pandas as pd
import spacy

# %% Load the model of the language to use for the pre-processing
nlp = spacy.load("en_core_web_lg")

# %% Load the data
