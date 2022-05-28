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

# %%
# Import libraries
#
import glob, os
import spacy
from rich.console import Console
from rich.table import Table

# %% Load the model of the language to use for the pre-processing
nlp = spacy.load("en_core_web_sm")

# %%
# Load the transcripts for a sample of commencement speeches
#
# browse target folder
fdr = "../sampleData/commencementSpeeches/corpus"
in_fs = glob.glob(os.path.join(fdr, "*.txt"))
# load the transcripts into a dictionary
speeches = {}
for in_f in in_fs:
    with open(in_f, "r") as f:
        key = int(in_f.split("_")[1].rstrip(".txt"))
        speeches[key] = f.read()
        del key

# %%
# create a Rich's table to print the output of the spaCy's pipeline
console = Console()
# defin table properties
table = Table(
    show_header=True,
    header_style="bold #2070b2",
    title="[bold] [#2070b2] A pre-processing NLP pipeline with spaCy[/#2070b2]",
)
# add columns
table.add_column("Token text")
table.add_column("Lemma")
table.add_column("POS")
table.add_column("DEP")
table.add_column("Alpha")
table.add_column("Stop")
# let's consider the first speech in the dictionary
doc = speeches[0]
# we retrieve the tokens' attributes and we add them to the table
for token in nlp(doc):
    table.add_row(
        token.text,
        token.lemma_,
        token.pos_,
        token.dep_,
        str(token.is_alpha),
        str(token.is_stop),
    )
# print the table
console.print(table)
