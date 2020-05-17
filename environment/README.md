Environment ― README
====================

This folder contains the information to setup a pure-Python- or Conda-based, NLP environment.

The file `requirements.txt` contains the set of Python libraries to install.

Pure-Python users do:

```{python}
pip install -r requirements.txt
```

Conda users do:

```{python}
conda install --file requirements.txt
```

Note for Conda users: pyLDAvis and Staza are not available via the official conda channel.

Stanza can be installed as follows:

```{python}
conda install -c stanfordnlp stanza
```

pyLDAvis can be installed as follows:

```{python}
conda install -c conda-forge pyldavis
```
