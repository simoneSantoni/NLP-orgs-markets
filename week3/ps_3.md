# Week 3, problem set

Here's the problem set due on June 9 (5:00 PM):

Consider the ['commencement speech corpus'](/Users/simone/githubRepos/applied-NLP-smm694/data/commencementSpeeches),
which contains the transcripts for twelve speeches given by public 
leaders at North-American universities.

Use pure spaCy and NumPy and/or SciPy and/or Pandas to carry out the
following tasks:

- pass the documents through the spaCy NLP pipeline (preferrably, use 
  the model of the language 'en_core_wb_lg')
- plot the frequency distribution for:
  - the unique words in the aggregated corpusof text (that is, the 
    concatenation of the twelve transcripts)
  - the lemmas included in the transformed, aggregated corpus of text
    (that is, the concatenation of the outcome of the spaCy pipeline
     applied over the twelve transcripts)
- use the word vectors included in the model of the language and a distance
  metric of your choice to assess the semantic similarity between pairs of documents. Tip:
  [SciPy implements a variety of distance metrics](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)
- visualize the semantic similarity between pairs of
  speeches. Tip: you may want to use some [Manifold Learning
  techniques](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold)
  from scikit-learn library. How about MDS or TSNE?

Let me stress that you can work on the problem set on your own or with your
teammates.

Send your document via email to simone.santoni.1@city.ac.uk
