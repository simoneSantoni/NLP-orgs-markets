# Week 2, problem set

Here's the problem set due on June 2 (5:00 PM):

Consider the ['commencement speech corpus'](/Users/simone/githubRepos/applied-NLP-smm694/data/commencementSpeeches),
which contains the transcripts for twelve speeches given by public 
leaders at North-American universities.

Use pure Python and/or NumPy and/or SciPy and/or Pandas to carry out the
following tasks:

- produce the vocabulary of unique words
- plot the frequency distribution of unique words in the aggregated corpus 
  of text (that is, the concatenation of the twelve transcripts)
- use the vocabulary of words included in the aggregate corpus
  to represent each transcript as a bag of words
- use the distance metric of your choice (e.g., cosine similarity) 
  to assess the semantic similarity between pairs of documents. Tip:
  [SciPy implements a variety of distance metrics](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)
- (optional point) visualize the semantic similarity between 
  speeches. Tip: you may want to use some [Manifold Learning
  techniques](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold)
  from scikit-learn library. How about MDS or TSNE?

Let me stress that you can work on the problem set on your own or with your
teammates.

Send your document via email to simone.santoni.1@city.ac.uk
