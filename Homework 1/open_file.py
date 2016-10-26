import gzip
import cPickle as pickle
f = gzip.open('C:/Users/yawen/Desktop/CSCI 5622 Machine Learning/Homework 1/mnist.pkl.gz','rb')
info = pickle.load(f)
print info #show file
