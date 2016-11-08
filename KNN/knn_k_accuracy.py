import argparse
from collections import Counter, defaultdict

import random
import numpy
from numpy import median
from sklearn.neighbors import BallTree

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()


class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k=5):
        """
        Creates a kNN instance
        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        # You can modify the constructor, but you shouldn't need to.
        # Do not use another data structure from anywhere else to
        # complete the assignment.

        self._kdtree = BallTree(x)
        self._y = y
        self._k = k

    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median of the majority labels (as implemented 
        in numpy).
        :param item_indices: The indices of the k nearest neighbors
        """
        assert len(item_indices) == self._k, "Did not get k inputs"

        # Finish this function to return the most common y label for
        # the given indices.  The current return value is a placeholder 
        # and definitely needs to be changed. 
        #
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html
        
        # Calculate the frequency for k nearest neighbors
        freq_list = Counter([self._y[i] for i in item_indices])
        freq_list_value = freq_list.values()

        # Calculate the max frequency and list all of them
        max_cnt = max(freq_list_value)
        total_list = freq_list_value.count(max_cnt)
        Majority_list = freq_list.most_common(total_list)   

        # Use median to calculate the y for 
        Majority_label = median(zip(*Majority_list)[0])
        
        return Majority_label

    def classify(self, example):
        """
        Given an example, classify the example.
        :param example: A representation of an example in the same
        format as training data
        """

        # Finish this function to find the k closest points, query the
        # majority function, and return the predicted label.
        # Again, the current return value is a placeholder 
        # and definitely needs to be changed. 
        
        # warning !!       
        item_distances, item_indices = self._kdtree.query(example.reshape(1, -1), self._k)

        return self.majority(item_indices[0])

    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrix for the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.
        :param test_x: Test data representation
        :param test_y: Test data answers
        """

        # Finish this function to build a dictionary with the
        # mislabeled examples.  You'll need to call the classify
        # function for each example.

        d = defaultdict(dict)
        data_index = 0

        for xx, yy in zip(test_x, test_y):
            data_index += 1
             
            true_label = yy

            # Call classify function
	    classfied_label = self.classify(xx)
            
            try:
                d[true_label][classfied_label] += 1
            except KeyError:
                d[true_label][classfied_label] = 1

        return d

    @staticmethod
    def accuracy(confusion_matrix):
        """
        Given a confusion matrix, compute the accuracy of the underlying classifier.
        """

        # You do not need to modify this function

        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)

        if total:
            return float(correct) / float(total)
        else:
            return 0.0


if __name__ == "__main__":


    
    Accuracy = [None] * 50
    
    for l in range(1, 51, 1):
        
        parser = argparse.ArgumentParser(description='KNN classifier options')
        # Set k
        parser.add_argument('--k', type=int, default=l, help="Number of nearest points to use")
        # Set Training Data limit
        parser.add_argument('--limit', type=int, default=-1, help="Restrict training to this many examples")
        args = parser.parse_args()

        data = Numbers("/home/yawen/Desktop/CSCI 5622 Machine Learning/Homework 1/mnist.pkl.gz")

        # You should not have to modify any of this code

        if args.limit > 0:
            print("k: %i" % l)
            knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit], args.k)
        else:
            knn = Knearest(data.train_x, data.train_y, args.k)
            print args.k

        confusion = knn.confusion_matrix(data.test_x, data.test_y)
        
        Accuracy[l - 1] = knn.accuracy(confusion)       
        print("Accuracy: %f" % knn.accuracy(confusion))

    numpy.savetxt('k_accuracy.txt', Accuracy)


