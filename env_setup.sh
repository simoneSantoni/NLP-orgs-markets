#----------------------------------------------------------------------------- 
#
# This shell script sets up the Python environment for the build process.
#
# Author : Simone Santoni, simone.santoni.1@city.ac.uk
# Date   : 2022-05-25
# Notes  : NaN
#
#----------------------------------------------------------------------------- 

# default channel libraries
conda install ipython jupyter numpy scipy matplotlib pandas gensim nltk

# community channel libraries
conda install -c conda-forge scikit-learn spacy tomotopy python-flair textblob

# pytorch has its own channel and comes with multiple packages 
conda install pytorch torchvision torchaudio -c pytorch