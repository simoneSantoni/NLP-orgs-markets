Final Course Project - README
=============================

Table of Contents
=================

<!-- vim-markdown-toc GFM -->

:
* [Scope](#scope)
* [Data](#data)
* [Deliverables](#deliverables)

<!-- vim-markdown-toc GFM -->

Scope
=====

For the **FCP** ― to be launched in week 6 ―, students are supposed:

1.  to prepare and analyze a real-world dataset containing
    press-releases, business reports, and financial analysts reports
    (all relevant will be made available in week 5);

2.  to use the main insights emerging from 1) to analyze the performance
    of British, publicly-listed companies in the aftermath of the 2016
    Brexit Referendum. A group of publicly listed companies based in
    France and Germany will offer the counterfactual data to estimate
    how British companies could have performed in case of no-leave.

FCP submissions will be evaluated on a rolling-based window and are due
by July 17 (8:00 PM London Time).

Here are some questions students may want to explore (just one):

+ do annual reports' contents style change after the Brexit
Referendum? Then, with what consequences for the economic-financial 
performance of companies?
+ does annual reports' emphasis on sustainable initiatives and CSR 
change after the Brexit Referendum? Then, with what consequences for the 
economic-financial performance of companies?
+ do new topics emerge in annual reports after the Brexit Referendum? Then, 
with what consequences for the economic-financial performance of companies?
+ is there any particular topic or set of topics that positively correlates
with economic and financial performance in the post Brexit Referendum period?
Then, why?


**!!!  Notes ¡¡¡** The above-displayed questions are just sample questions. Students are
free to pick-up a different one.

Concerning the data preparation/transformation/analysis workflow, students may
want to consider the possible pathways:

+ NLP pipeline -> topic modeling -> save document-to-topic probabilities -> use
document-to-topic probabilities as features in the context of some statistical
or ML analysis
+ NLP pipeline -> doc2vec -> use document similarity scores as features in the 
context of some statistical or ML analysis
+ NLP pipeline -> isolate concepts/words of interest (i.e., seed words; e.g. 
'CSR') -> assess how frequent those concepts are in documents across companies 
and times -> use frequencies as features in the context of some statistical 
analysis of ML analysis.

Tip to model the economic-financial performance of UK companies relative to 
German and French companies: use the so-called Difference-in-Differences (DiD)
design (see Sieweke & Santoni, 2020, page: ). Here's the intuition of the DiD:
let's assume your data contain longitudinal data on two types of companies,
namely, treated and control companies. Treated companies only experience an 
environmental change, i.e., the increase in institutional uncertainty associated
with the 'leave' decision. This translates in a dataset that looks like this:

| Company | Year | Country | Treated (d) | Post-Brexit Year (t) | y   | x   | z   |
|---------|------|---------|-------------|----------------------|-----|-----|-----|
| A       | 2015 | UK      | Yes         | No                   | ... | ... | ... |
| A       | 2016 | UK      | Yes         | No                   | ... | ... | ... |
| A       | 2017 | UK      | Yes         | Yes                  | ... | ... | ... |
| B       | 2015 | FR      | No          | No                   | ... | ... | ... |
| B       | 2016 | FR      | No          | No                   | ... | ... | ... |
| B       | 2017 | FR      | No          | Yes                  | ... | ... | ... |

The economic-financial performance (y) can then be modeled as:

y = α + β1 * x + β2 * z + γ * d + δ * t + ζ * d * t + ε 

The estimated quantity ζ captures the change in performance of UK companies
(i.e., treated units) relative to non-UK companies (i.e., control units). In
fact, the interaction between the binary variables d and t is equal to 1 if and
only if the observations pertain to a UK company in a post-Brexit referendum
year.

Well, it would be extremely interesting to understand how ζ changes across UK
companies and why. I wonder if NLP can help students to address this problem.

Data
====

The data for the project contains:

+ business press corpus
+ company-level data
  - population of companies
  - economic & financials (both short- and long-term)
  + annual reports

Business Press Corpus
---------------------

The business press corpus contains newspaper articles from the following 
newspapers: Financial Times; Telegraph; The Guardian; The Times.

Key features of the corpus:

* retrieved articles contain the keywords 'sustainability' an/or 'CSR'. The
choice for the keywords is motivated by the prior literature and empirical 
evidence showing 'good' companies are better equipped to navigate turbulent
competitive landscapes ― in other words, investments in sustainable initiatives
and/or CSR are supposed to create/sustain a positive corporate identity, which,
in turn, can act as a buffer protecting companies from institutional and market
uncertainty;
+ the **timespan** is April 23, 2016 - August 23, 2016 (that is, a  four month 
window around the Brexit Referendum of June 23, 2016).

The business press corpus contains the following data tables:

+ `pr__sources.csv`: dictionary of newspapers
+ `pr__sus_attr`: attributes of individual newspapers (long format, columns are
self-explanatory)
+ `pr__sus_docs`: text corpus of the articles (linked with `pr__sus_attr` via
the field `article`)

Company-level data
------------------

### Population of companies

The file `companies.csv` contains a list of companies along with the reference
country/stock market.

### Economic & financials

There are two files containing economic & financials:

+ `financials__long_term`: 
+ `financials__short_term`: 

### Annual reports

Annual reports were collected from the database Filing Experts (accessible via
the portal of City's Library). Note: reports are not available in certain years
― mainly, this is due to M&As/restructuring that change the legal entity
behind a company.


Deliverables
============

Possible ways to deliver the projects ()


