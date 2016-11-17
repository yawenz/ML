from csv import DictReader, DictWriter

import numpy as np
from numpy import array
import random
import nltk
import re

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import SGDClassifier

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'
kTROPE_FIELD = 'trope'
kPAGE_FIELD = 'page'

class Featurizer:
    def __init__(self):
        self.vectorizer = FeatureUnion( 
        [       
                ('bag of words', 
                  Pipeline([('extract_field', FunctionTransformer(lambda x: x[0], validate = False)),
                            ('tfid', TfidfVectorizer(stop_words = 'english'))])),              
                ('page in words',
                  Pipeline([('extract_field', FunctionTransformer(lambda x: [x[0], x[2]], validate = False)), 
                            ('page', PageTransformer())])),
                ('type of trope', 
                  Pipeline([('extract_field', FunctionTransformer(lambda x: x[1], validate = False)),
                            ('trope_count', CountVectorizer())]))
                
        ])
    
    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))

class PageTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, examples):
        # return self and nothing else 
        return self
    
    def transform(self, examples):
        
        import numpy as np 
        from scipy.sparse import csr_matrix
            
        # Initiaize matrix 
        X = np.zeros((len(examples[0]), 1))
        
        # Loop over examples and count letters 
        for ii, x in enumerate(examples[0]):            
            sentence_word = nltk.word_tokenize(x)
            page = re.sub(r'([A-Z])', r' \1', examples[1][ii])           
            page_word = nltk.word_tokenize(page)
            common_word = set(sentence_word).intersection(page_word)
            # Remove "The" from common_word
            if 'The' in common_word:
                common_word.remove('The')
                
            X[ii,:] = len(common_word)
        
        X = preprocessing.normalize(X, norm='l2')
        return csr_matrix(X)     

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("/Users/yawen/Desktop/CSCI 5622 Machine Learning/Homework 6/feature_engineering/data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("/Users/yawen/Desktop/CSCI 5622 Machine Learning/Homework 6/feature_engineering/data/spoilers/test.csv", 'r')))

    feat = Featurizer()

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    print("Label set: %s" % str(labels))

    x_train = feat.train_feature([[x[kTEXT_FIELD] for x in train], [x[kTROPE_FIELD] for x in train], [x[kPAGE_FIELD] for x in train]])
    x_test = feat.test_feature([[x[kTEXT_FIELD] for x in test], [x[kTROPE_FIELD] for x in test], [x[kPAGE_FIELD] for x in test]])

    y_train = array(list(labels.index(x[kTARGET_FIELD])
            for x in train))

    print(len(train), len(y_train))
    print(set(y_train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    #feat.show_top10(lr, labels)

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'spoiler': labels[pp]}
        o.writerow(d)
