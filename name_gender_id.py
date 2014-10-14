'''
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
'''
import sys
import random
from sklearn.metrics import roc_curve, auc
import nltk
from nltk.tag.stanford import NERTagger

# Given a sentence, produce a list of peoples' full names
class Name_Identifier():

    def __init__(self, tagger='Stanford'):
        self.tagger = tagger

    def get_names(self, sentence):
        # Use NLTK Tagger
        if self.tagger == 'NLTK':
            tokens = nltk.tokenize.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(tokens)
            ner_tags = nltk.ne_chunk(pos_tags)

        # Use Stanford NER Tagger instead of NLTK default
        elif self.tagger == 'Stanford':
            st = NERTagger('/usr/share/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
                       '/usr/share/stanford-ner/stanford-ner.jar') 
            ner_tags = st.tag(sentence.split())

        return self.get_names_from_tags(ner_tags)

    def get_names_from_tags(self, ner_tags):
        names = []
        if self.tagger == 'Stanford':
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

        elif self.tagger == 'NLTK':
            for subtree in ner_tags.subtrees(filter=lambda t: t.node == 'PERSON'):
                name = ''
                for leaf in subtree.leaves():
                    name += leaf[0] + ' '
                names.append(name)
        return names


# Given name, classify the name's gender as male or female
class Gender_Predictor():
    
    def __init__(self):
        self.classifier = None
        self.get_train_and_test_data()
        self.build_classifier()

    def get_train_and_test_data(self, train_split_pct=.8):
        names_male = [name for name in nltk.corpus.names.words('male.txt')]
        names_female = [name for name in nltk.corpus.names.words('female.txt')]
        names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])

        male_set = set(names_male)
        female_set = set(names_female)
        male_set -= female_set
        female_set -= male_set

        self.male_set = male_set
        self.female_set = female_set
        random.shuffle(names)

        self.feature_sets = [(name_features(name), gender) for (name, gender) in names]
        self.train_set, self.test_set = feature_sets[1000:], featuresets[:1000]


    def get_name_features(self):
        name = self.first_name.lower()
        return {
            'last_letter' : name[-1],
            'last_two' : name[-2:],
            'last_is_vowel' : (name[-1] in 'aeiouy')
        }

    def build_classifier(self):
        # Naive Bayes Classifier
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)


    def predict_gender(self, name):
        name_features = self.get_name_features(name) 
        return self.classifier.classify(name_features)


if __name__ == '__main__':
    #sentence = ' '.join(sys.argv[1:])
    sentence = 'Some economists have responded positively to Bitcoin, including Francois R. Velde'
    names = Name_Identifier('NLTK').get_names(sentence)
    names = Name_Identifier('Stanford').get_names(sentence)
    GP = Gender_Predictor()
    GP.predict_gender(name)
