# Hackathon #1

![](images/lebron.jpeg)

## Intro

What are the semantic features associated with successful leadership? 
In other words, how do collectives represent effective leaders using 
natural language?

Despite public leaders exert disproportionate impact on organizations,
societies, and institutions, there's scant knowledge on the factors associated
with effective leadership. Bridging such a gap is key to promote the development
of impactful CEOs, politicians, or even professional sport superstars.

With this first hackathon, you'll contribute to get a closer understanding 
of the semantic features associated with effective leadership. Particularly,
you're required to go through __three steps__:

1. Load [pre-trained word embeddings](https://code.google.com/archive/p/word2vec/)
2. Load [survey data](https://github.com/simoneSantoni/applied-NLP-smm694/blob/master/data/leadershipEffectiveness/prolific_data.csv) containing human beings' ratings expressing the effectiveness
   of 299 public leaders (e.g., Simone rates the effectiveness of 
   Harry Kane as a leader)
3. Create a statistical model or a Machine Learning model that reveals the 
   mapping of semantic features onto leadership effectiveness ratings

## Documentation

### Step 1: Load pre-trained word embeddings

Download and load the $\texttt{word2vec}$ embeddings trained onto the 
[Google News corpus](https://code.google.com/archive/p/word2vec/).
That's easy to accomplish － for example, you may want to use Gensim:

```{python}
import gensim.downloader as api
model = api.load("word2vec-google-news-300")
word_vectors = model[""]
```

### Step 2: Load the leadership effectiveness scores

These data have been gathered from 210 online participants (99 female, 105 male,
6 other; mean age = 32) recruited from Prolific Academic. Overall, these
participants rated 299 target individuals included in the [Pantheon
dataset](https://pantheon.world/data/datasets) in terms of effective leadership.
Specifically, each participant was presented with the names of 50 individuals
randomly selected from the target set of 299, and asked to rate how effective of
a leader they thought each target individual is, was, or would be, on a 0-100
scale.

As with Offermann et al. study (1994), participants were explicitly instructed to
provide their ratings based on their personal belief of what makes a good
leader. They were allowed to select an “I don’t know [this person]” response
option whenever they did not recognize a target individual’s name. The target
names were presented on separate pages, and in random order. Our participants
were all US citizens with a Prolific Academic approval rating of 90% or higher.
They were overwhelmingly Caucasian (160 Caucasian, 50 non-Caucasian) and nearly
half identified as Democrat (100 Democrat, 28 Republican, 82 other).

The data table contains 210 responses (i.e., 210 rows) regarding 299 leaders (each 
column is associated with a leader's name).

Pandas is a great option to load (and manipulate) the data:

```{python}
import pandas as pd
df = pd.read_csv("../data/leadershipEffectiveness/prolific_data.csv")
```

## Your task (Step 3)

Use the statistical or ML framework of your choice to reveal the semantic
features that are more closely associated with or best predict leadership
effectiveness.  Admitted languages for the statistical or ML analysis: Julia,
Python, R, Rust, or Stata.

## Deliverables and deadline

By June 9 (5:00 PM) submit the code for your project and a 5-frame slideshow
illustrating: i) the process for your analysis; ii) the main result you achieve, 
i.e., the extent to which you can explain of predict leadership effectiveness
using word embeddings.