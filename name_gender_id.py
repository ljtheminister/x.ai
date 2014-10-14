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


class Name_Identifier():

    def __init__(self, sentence, tagger='Stanford'):
        self.sentence = sentence
        self.names = None
        self.tagger = tagger

    def get_names(self):
        # Use NLTK Tagger
        if self.tagger == 'NLTK':
            tokens = nltk.tokenize.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(tokens)
            ner_tags = nltk.ne_chunk(pos_tags)
            self.get_names_from_tags(ner_tags, 'NLTK')

        # Use Stanford NER Tagger instead of NLTK default
        elif self.tagger == 'Stanford':
            st = NERTagger('/usr/share/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
                       '/usr/share/stanford-ner/stanford-ner.jar') 
            ner_tags = st.tag(self.sentence.split())
            self.get_names_from_tags(ner_tags, 'Stanford')

    def get_names_from_tags(self, ner_tags, tag_type='Stanford'):
        names = []
        if tag_type == 'Stanford':
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

        elif tag_type == 'NLTK':
            for subtree in ner_tags.subtrees(filter=lambda t: t.node == 'PERSON'):
                name = ''
                for leaf in subtree.leaves():
                    name += leaf[0] + ' '
                names.append(name)
        return names


n = Name_Identifier(sentence, 'NLTK')
n.get_names()

class Gender_Predictor():
    
    def __init__(self, name):
        self.name = name
        self.first_name = name.split()[0]
        self.__build_classifier()


    def build_classifier(self):

        self.male_set = male_set
        self.female_set = female_set

# Naive Bayes Classifier
names_male = [name for name in nltk.corpus.names.words('male.txt')]
names_female = [name for name in nltk.corpus.names.words('female.txt')]
names = ([(name, 'male') for name in names.words('male.txt')] +
         [(name, 'female') for name in names.words('female.txt')])


male_set = set(names_male)
female_set = set(names_female)
male_set -= female_set
female_set -= male_set

random.shuffle(names)

feature_sets = [(gender_features(name), gender) for (name, gender) in names]
train_set, test_set = feature_sets[1000:], featuresets[:1000]

classifier_NB = nltk.NaiveBayesClassifier.train(train_set)

def _nameFeatures(self):
    name = self.first_name.lower()
    return {
        'last_letter' : name[-1],
        'last_two' : name[-2:],
        'last_is_vowel' : (name[-1] in 'aeiouy')
    }






if __name__ == '__main__':
    sentence = 'Some economists have responded positively to Bitcoin, including Francois R. Velde'
    names = Name_Identifier(sentence)
    genders = Gender_Predictor(names)
