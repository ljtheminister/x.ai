import sys
import os
from collections import defaultdict
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.parse.dependencygraph import DependencyGraph
from nltk.stem import porter
from nltk.tokenize import RegexpTokenizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk

from nltk.corpus import names
import random

names_male = [name for name in names.words('male.txt')]
names_female = [name for name in names.words('female.txt')]

names = ([(name, 'male') for name in names.words('male.txt')] +
         [(name, 'female') for name in names.words('female.txt')])

train_set, test_set = feat


from nltk import NaiveBayesClassifier, classify



def _nameFeatures(self,name):
    name = name.lower()
    return {
        'last_letter' : name[-1],
        'last_two' : name[-2:],
        'last_is_vowel' : (name[-1] in 'aeiouy')
    }


'''
What is performance of NER algorithm?
'''



class Name_Identifier():

    def init


def get_names(sentence):
tokens = nltk.tokenize.word_tokenize(sentence)
pos_tags = nltk.pos_tag(tokens)
sent = nltk.ne_chunk(pos)






class Gender_Predictor():


text = """
Some economists have responded positively to Bitcoin, including 
Francois R. Velde, senior economist of the Federal Reserve in Chicago 
who described it as "an elegant solution to the problem of creating a 
digital currency." In November 2013 Richard Branson announced that 
Virgin Galactic would accept Bitcoin as payment, saying that he had invested 
in Bitcoin and found it "fascinating how a whole new global currency 
has been created", encouraging others to also invest in Bitcoin.
Other economists commenting on Bitcoin have been critical. 
Economist Paul Krugman has suggested that the structure of the currency 
incentivizes hoarding and that its value derives from the expectation that 
others will accept it as payment. Economist Larry Summers has expressed 
a "wait and see" attitude when it comes to Bitcoin. Nick Colas, a market 
strategist for ConvergEx Group, has remarked on the effect of increasing 
use of Bitcoin and its restricted supply, noting, "When incremental 
adoption meets relatively fixed supply, it should be no surprise that 
prices go up. And that’s exactly what is happening to BTC prices."
"""
