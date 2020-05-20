Text and machines
=================

How do machines think about text?

+ answering this question is key to understand how to search for, retrieve, and
    edit patterns of text
+ regular expressions (regex) offer a possible framework to instruct machines to
    represent natural language as a mixture of formal expressions (next week,
    we'll go deeper into the rabbit hole)


Regex as a pipeline
===================

+ formally, a regular expression is an "algebraic notation for characterizing a
    set of strings" (Jurafski and Martin, 2008)
+ a key features of regex as and algebraic notation is the possibility to
    express fairly complex strings in relatively simple terms
+ regex can also be thought as a pipeline that connect the world of natural
    language (the one we, humans, speak) with the world of formal language (that
    is, the language f machines)

Regex in Python
===============

+ Python implements the so-called 'extended regular expressions' parser
+ caveat: if you use other programming languages (that rely on different 
    parsersa), your expressions could not be portable
+ the [`re`](https://docs.python.org/3/library/re.html) module for Python 
    has stellar quality documentation

Learning regex by (simple) examples
===================================

Consider these sample sentences:

```{bash}
'Apple laptops are for starbucksers'

'Did an apple really fall on Isaac Newton's head?'

'How do I fix my Apple MacBookPro? I spilled apple juice over the keyboard'

'The brand-new MacBookPro13-in costs £ 1,500'

'Each glass of apple juice takes 3 apples'

'run ran run'

'The CEO of alfa announced the acquisition of beta. The reaction of
stakeholders has been positive.'
```
Let's see regex in action:

+ regex are case sensitive
+ using the `^` symbol to negate a pattern
+ 'the preceding character or nothing'
+ between `a` and `c`
+ path repetition
+ one or more occurrences of the immediately preceding character
+ using the `^` symbol to match the start of a line
+ disjunctions 
+ grouping

Popular operators
=================

| RE | Expansion    | Match              | First Matches     |
|----|--------------|--------------------|-------------------|
| d  | [0-9]        | any digit          | Party?of?5        |
| \D | [ˆ0-9]       | any non-digit      | Blue?moon         |
| \w | [a-zA-Z0-9_] | any alphanumeric   | /underscore Daiyu |
| \W | [ˆ\w]        | a non-alphanumeric | !!!!              |
| \s | [?\r\t\n\f]  | whitespace         | (space, tab)      |
| \S | [ˆ\s]        | Non-whitespace     | in?Concord        |

Source: Jurafski and Martin, 2008

Words, text corpus, and text corpora
====================================

+ words are the fundamental unit of observations
+ words that appear in a same document constitute a corpus of text
+ collections of documents constitute text corpora

Text normalization
==================

Before almost any natural language processing of a text, the text has to be
normalized. At least three tasks are commonly applied as part of any
normalization process:

1. Tokenizing (segmenting) words 
2. Normalizing word formats 
3. Segmenting sentences
