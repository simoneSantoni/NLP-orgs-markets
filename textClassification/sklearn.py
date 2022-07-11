# /usr/env/bin python3
# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------
    sklearn.py  |  Text Classification with Scikit-Learn and doc-to-topic
                |  probabilities
------------------------------------------------------------------------------

Author   : Simone Santoni, simone.santoni.1@city.ac.uk 

Synopsis : this script uses scikit-learn to train a text classifier fed with 
           document-to-topic probabilities from an LDA model. The dataset 
           contains 20,491 rating-review pairs from Tripadvisor.

Notes    : none

"""

# %%
# load libraries
import numpy as np
import pandas as pd


# %%
# 