Environment ― README
====================

This folder contains the information to setup a pure-Python or Conda-based NLP
environment.

The file `requirements.txt` contains the set of Python libraries to install.

Pure-Python[1] users do:

```{python}
pip install -r requirements.txt
```

Conda[2] users do:

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

--
Notes

[1] A new Python environment can be created using the `venv` module as follows: `python -m
venv create my_env`

[2] A new Conda environment can be created as follows: `conda create -n my_env python=3.x`
