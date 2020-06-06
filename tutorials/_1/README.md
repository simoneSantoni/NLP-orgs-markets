Tutorial 1 ― README
===================

This tutorial deals with the following topics:

+ NLP pipeline

**Stanza is required** to reproduce the tutorial.

Why Stanza? Mainly, there are four arguments behind the choice to use Stanza 
instead of competing libraries such as spaCy: 
                                                                        
1. Stanza builds upon PyTorch. Implication: present a GPU-enabled       
   machine, Stanza's performance increases
                                                                       
2 [Stanza `neural` pipeline is more accurate than spaCy's one][1]
   (but slower). In the spaCy universe, there's a library that allows
   [to use Stanza as a spaCy pipeline][2]

3. availability of multi-word tokens, which allow for more accurate 
   and nuanced representation of the linkages connecting a token and its
   underlying syntactic words (or vice versa)
                                                                        
4. seamless integration with Stanford CoreNLP [1], a very established, 
   mature project offering NLP capabilities and state of the are
   text analytics (APIs available for several languages; CoreNLP can 
   also run as a web service) 
                                                                        
[1]: https://stanfordnlp.github.io/CoreNLP/                              
                                                                    
[2]: https://spacy.io/universe/project/spacy-stanza                      
