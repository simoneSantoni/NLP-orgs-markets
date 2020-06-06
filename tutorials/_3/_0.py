# !/usr/bin/env python3

"""
Docstring
-------------------------------------------------------------------------------
    _0.py    |    example of Minimum Edit Distance application
-------------------------------------------------------------------------------

Author: Simone Santoni, simone.santoni.1@city.ac.uk

Notes: Appreciating the approximate or phonetic distance between two string
       is a common task when it comes to manipulate strings. The jellyfish
       library for Python implements several distance metrics.

       One of the metrics available in the jellyfish library is the 
       Levenshtein Distance / Minimum Edit Distance, the number of insertions, 
       deletions, and substitutions required to change one word to another.

       One possible application
       is identifying succesful patch/code submissions on the part of 
       developers or data scientists working in a team setting 
       (e.g., GitHub teams). For example, the members of the Linux Kernel 
       community exchange their code using the mailing list of the project, 
       while very few members have the power to accept (i.e., to commit)
       a patch/code submissions. This raises concerns on who contributed
       to what and to what extent.

       This script applies the Levensthein Distance/Minimum Edit Distance to
       appreciate the similarity between some code included in a fictional email
       and the commit that is visible in a fictional, reference repository.
       

       [1] https://github.com/jamesturk/jellyfish

"""

# %% load library
import jellyfish as je


# %% fake data

# user 'i' shares some code via email

mail = """
Hi there-
over the last few days, I've been working on task ABC. The below-displayed
code solves the problem:
> for i in np.arange(0, 10, 1):
>    print(i)
"""

# the owner of the repository accepts i's proposal without revisions

commit_a = """
Hi Homer-
fine, your code has been committed to the repo.
Saludos,
Mr. Burns
> for i in np.arange(0, 10, 1):
>    print(i)
"""

# the owner of the repository accepts i's proposal with minor revisions

commit_b = """
Hi Homer-
your code has been committed to the repo with minor corrections.
Saludos,
Chief Clancy Wiggum 
> for i in range(10):
>    print(i)
"""


# %% compare and contrast strings

# function to extract code from strings
def get_code(_s):
    '''
    : argument: string (containing natural language and code)
    : return  : string (containing code only)
    '''
    _split = _s.split('\n')
    _code = [line.strip('>').strip() 
             for line in _split if line.startswith('> ')]
    return ' '.join(_code)


# transform string to focus on code
s1 = get_code(mail)
s2, s3 = get_code(commit_a), get_code(commit_b)

# compute Levensthein Distance
s1_s2 = je.levenshtein_distance(s1, s2)
s1_s3 = je.levenshtein_distance(s1, s3)

# print results
print("""Email - commit a distance: {}
Email - commit b distance: {}""".format(s1_s2, s1_s3))

'''
Conclusion:

+ the comparison of s1 and s2 clearly indicates Homer contributed
  to the project as per the commit that is visible in the colalborative repo.

+ the comparison of s1 and s3 raises some concerns ― if you don't observe
  any Homer's commits to the collaborative repo (because he doesn't own
  the repo) it could be hard to affirm Homer made a contribution.
'''
