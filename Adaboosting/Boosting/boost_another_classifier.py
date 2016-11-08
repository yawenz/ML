import argparse
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.base import clone 
import matplotlib.pyplot as plt
import csv


np.random.seed(1234)

class FoursAndNines:
    """
    Class to store MNIST data
    """

    def __init__(self, location):

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')

        # Split the data set 
        train_set, valid_set, test_set = cPickle.load(f)

        # Extract only 4's and 9's for training set 
        self.x_train = train_set[0][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0],:]
        self.y_train = train_set[1][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0]]
        self.y_train = np.array([1 if y == 9 else -1 for y in self.y_train])
        
        # Shuffle the training data 
        shuff = np.arange(self.x_train.shape[0])
        np.random.shuffle(shuff)
        self.x_train = self.x_train[shuff,:]
        self.y_train = self.y_train[shuff]

        # Extract only 4's and 9's for validation set 
        self.x_valid = valid_set[0][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0],:]
        self.y_valid = valid_set[1][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0]]
        self.y_valid = np.array([1 if y == 9 else -1 for y in self.y_valid])
        
        # Extract only 4's and 9's for test set 
        self.x_test  = test_set[0][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0],:]
        self.y_test  = test_set[1][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0]]
        self.y_test = np.array([1 if y == 9 else -1 for y in self.y_test])
        
        f.close()

class AdaBoost:
    def __init__(self, n_learners=20, base=DecisionTreeClassifier(max_depth=1)):
        """
        Create a new adaboost classifier.
        
        Args:
            n_learners (int, optional): Number of weak learners in classifier.
            base (BaseEstimator, optional): Your general weak learner 

        Attributes:
            base (estimator): Your general weak learner 
            n_learners (int): Number of weak learners in classifier.
            alpha (ndarray): Coefficients on weak learners. 
            learners (list): List of weak learner instances. 
        """
        
        self.n_learners = n_learners
        self.base = base
        print self.base
        self.alpha = np.zeros(self.n_learners)
        self.learners = []
        
    def fit(self, X_train, y_train):
        """
        Train AdaBoost classifier on data. Sets alphas and learners. 
        
        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data 
        """
        # define w_i as 1/m
        w = np.ones(len(y_train)) / len(y_train)
        print w
        # define err_k as 0
        err = np.zeros(self.n_learners)
        print err
        
        for k in range(self.n_learners):
            
            # clone the parameter for classification
            h = clone(self.base)
            self.learners.append(h)

            h.fit(X_train, y_train, sample_weight = w)
            h_predict = h.predict(X_train)
            print h_predict
            
            # compute weighted error err_k
            for i in range(len(y_train)):
                if h_predict[i] != y_train[i]:
                    err[k] = err[k] + float(w[i]) / sum(w)
            
            # compute alpha_k
            self.alpha[k] = 0.5 * np.log(float(1 - err[k]) / err[k])
            print "alpha", k, self.alpha[k]
            
            # update w_i
            for i in range(len(y_train)):
                if h_predict[i] == y_train[i]:
                    w[i] = w[i] * np.exp(-self.alpha[k])
                else:
                    w[i] = w[i] * np.exp(self.alpha[k])
            # normalize w
            w = w / sum(w)
            print w
            
    def predict(self, X):
        """
        Adaboost prediction for new data X.
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            
        Returns: 
            [n_samples] ndarray of predicted labels {-1,1}
        """
        
        pred = np.zeros(X.shape[0])
        for k in range(self.n_learners):
            pred = pred + self.alpha[k] * self.learners[k].predict(X)
        pred = np.sign(pred)
            
        return pred
    
    def score(self, X, y):
        """
        Computes prediction accuracy of classifier.  
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            y (ndarray): [n_samples] ndarray of true labels  
            
        Returns: 
            Prediction accuracy (between 0.0 and 1.0).
        """
        score = 0
        pred_all = self.predict(X)
        for j in range(len(y)):
            if pred_all[j] == y[j]:
                score = score + 1
        score = float(score) / len(y)
                    
        return score
    
    def staged_score(self, X, y):
        """
        Computes the ensemble score after each iteration of boosting 
        for monitoring purposes, such as to determine the score on a 
        test set after each boost.
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            y (ndarray): [n_samples] ndarray of true labels  
            
        Returns: 
            [n_learners] ndarray of scores 
        """
        
        stage_score = np.zeros(self.n_learners)
        pred_k = np.zeros(len(y))
        
        for k in range(self.n_learners):
            pred_k = pred_k + self.alpha[k] * self.learners[k].predict(X)
            pred_k_sign = np.sign(pred_k)
            for i in range(len(y)):
                if pred_k_sign[i] == y[i]:
                    stage_score[k] = stage_score[k] + 1
            stage_score[k] = float(stage_score[k]) / len(y)
            
        return  stage_score


def mnist_digit_show(flatimage, outname=None):

	import matplotlib.pyplot as plt

	image = np.reshape(flatimage, (-1,28))

	plt.matshow(image, cmap=plt.cm.binary)
	plt.xticks([])
	plt.yticks([])
	if outname: 
	    plt.savefig(outname)
	else:
	    plt.show()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='AdaBoost classifier options')
	parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
	parser.add_argument('--n_learners', type=int, default=50,
                        help="Number of weak learners to use in boosting")
	args = parser.parse_args()

	data = FoursAndNines("/Users/yawen/Desktop/CSCI 5622 Machine Learning/Homework 4/Boosting/data/mnist.pkl.gz")

        # An example of how your classifier might be called 
	clf = AdaBoost(n_learners=300, base=Perceptron(n_iter = 3))
	clf.fit(data.x_train, data.y_train)

	# score_train = clf.score(data.x_train, data.y_train)
	stage_score_train = clf.staged_score(data.x_train, data.y_train)
	print stage_score_train
	# score_test = clf.score(data.x_test, data.y_test)
	stage_score_test = clf.staged_score(data.x_test, data.y_test)
	print stage_score_test

	stage_score_merge = zip(stage_score_train, stage_score_test)
	print stage_score_merge
	with open("/Users/yawen/Desktop/CSCI 5622 Machine Learning/Homework 4/Boosting/data/output_perceptron_300_3.csv", "wb") as f:
		writer = csv.writer(f)
		writer.writerows(stage_score_merge)

