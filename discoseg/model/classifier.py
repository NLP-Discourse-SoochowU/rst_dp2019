## classifier.py
## Author: Yangfeng Ji
## Date: 05-03-2015
## Time-stamp: <yangfeng 05/03/2015 09:27:27>

"""
A classification edu_repre_model for discourse segmentation
"""

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import _pickle as cPickle
import pickle
import gzip


class Classifier(object):
    def __init__(self, C=1.0, penalty='l2', loss='squared_hinge'):
        """ Initialization
        """
        self.C = C
        self.penalty = penalty
        self.loss = loss
        self.tol = 1e-7
        self.clf = None

    def train(self, data, labels, devdata=None, devlabels=None):
        """ Training
        """
        self.clf = LinearSVC(C=self.C, penalty=self.penalty,
                             loss=self.loss)
        self.clf.fit(data, labels)
        predlabels = self.clf.predict(data)
        acc = accuracy_score(labels, predlabels)
        print('Training Accuracy: {}'.format(acc))
        if devdata is not None:
            devpredlabels = self.clf.predict(devdata)
            devacc = accuracy_score(devlabels, devpredlabels)
            print('Dev Accuracy: {}'.format(devacc))

    def predict(self, data):
        """
        """
        predlabels = self.clf.predict(data)
        return predlabels

    def savemodel(self, fmodel):
        """
        """
        print('Save edu_repre_model into: {}'.format(fmodel))
        if not fmodel.endswith('.pickle.gz'):
            fmodel += '.pickle.gz'
        with gzip.open(fmodel, 'w') as fout:
            cPickle.dump(self.clf, fout)

    def loadmodel(self, fmodel):
        """
        """
        print("Load edu_repre_model from: {}".format(fmodel))
        with gzip.open(fmodel, 'rb') as fin:
            u = pickle._Unpickler(fin)
            u.encoding = 'latin1'
            self.clf = u.load()
            # self.clf = cPickle.load(fin)
