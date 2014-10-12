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
from nltk import NaiveBayesClassifier, classify
from nltk.corpus import names
import random
from sklearn.metrics import roc_curve, auc


from nltk.tag.stanford import NERTagger

names_male = [name for name in names.words('male.txt')]
names_female = [name for name in names.words('female.txt')]

names = ([(name, 'male') for name in names.words('male.txt')] +
         [(name, 'female') for name in names.words('female.txt')])


male_set = set(names_male)
female_set = set(names_female)
male_set -= female_set
female_set -= male_set

random.shuffle(names)
train_set, test_set = names[1000:], names[:1000]

def get_names(sentence):
    names = []
    '''
    tokens = nltk.tokenize.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    sent = nltk.ne_chunk(pos_tags)
    '''
    st = NERTagger('/usr/share/stanford-ner/classifiers/all.3class.distsim.crf.ser.gz',
               '/usr/share/stanford-ner/stanford-ner.jar') 
    ner_tags = st.tag(sentence)


def get_full_names(ner_tags):
    names = []
    name = None
    for i in xrange(len(ner_tags)):
        ne, ne_tag = ner_tags[i] 

        if ne_tag == 'PERSON':
            if (name):
                name += ' ' + ne
            else:
                name = ne

        else:
            if (name):
                names.append(name)
                name = None
    return names    



def name_features(name):
    name = name.lower()
    return {
        'last_letter' : name[-1],
        'last_two' : name[-2:],
        'last_is_vowel' : (name[-1] in 'aeiouy')
    }








class Name_Identifier():

    def __init__(self, sentence):
        self.sentence = sentence

    def get_names(self):
        ''' 
        tokens = nltk.tokenize.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokens)
        sent = nltk.ne_chunk(pos_tags)
        '''



class Gender_Predictor():
    
    def __init__(self, name):
        self.name = name
        self.first_name = name.split()[0]

    def _nameFeatures(self):
        name = self.first_name.lower()
        return {
            'last_letter' : name[-1],
            'last_two' : name[-2:],
            'last_is_vowel' : (name[-1] in 'aeiouy')
        }


    # Naive Bayes Classifier

from nltk.tag.stanford import NERTagger

st = NERTagger('/usr/share/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
               '/usr/share/stanford-ner/stanford-ner.jar') 

st.tag(sentence)


if __name__ == '__main__':
    sentence = 'Some economists have responded positively to Bitcoin, including Francois R. Velde'
    names = Name_Identifier(sentence)
    genders = Gender_Predictor(names)


for name in names:
    print name, Gender_Predictor(name)

# need to break down text into sentences

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

