import numpy as np

import sys
sys.path.append('../../')

from Util.util.data.DataPrep import *
from Util.util.math.MathUtil import *

class RNN(object):


    def __init__(self, hidden_layer_dim, input_dim, output_dim):
        self.hidden_layer_dim = hidden_layer_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Random Initialization of Parameters:
        self.U = np.random.uniform(-np.sqrt(1./self.input_dim), np.sqrt(1./self.input_dim), (self.hidden_layer_dim, self.input_dim))
        self.W = np.random.uniform(-np.sqrt(1./self.hidden_layer_dim), np.sqrt(1./self.hidden_layer_dim), (self.hidden_layer_dim, self.hidden_layer_dim))
        self.V = np.random.uniform(-np.sqrt(1./self.hidden_layer_dim), np.sqrt(1./self.hidden_layer_dim), (self.output_dim,self.hidden_layer_dim))
        self.S = []



    def step(self,Xz):
        T = len(Xz)
        self.S = np.zeros((T + 1, self.hidden_layer_dim))
        self.S[-1] = np.zeros(self.hidden_layer_dim)

        o = np.zeros((len(Xz),self.output_dim))
        for t in np.arange(T):
            self.S[t] = np.tanh(self.U.dot(Xz[t]) + self.W.dot(self.S[t-1]) )
            o[t] = MathUtil.softmax(self.V.dot(self.S[t]))

        return o

    def predict(self,input_sequence):
        O = self.step(input_sequence)
        return np.argmax(O,axis=1)

    def calculate_loss_OneCorrectClass(self,X,Y,O):
        total = 0
        for i in np.arange(len(Y)):
            # X[i] ---> Y[i]
            # X[i] ----> O[i]
            output_probabilities_of_correct_classes = O[i][np.arange(len(Y[i])),np.argmax(Y[i],axis=1)]
            yz = Y[i][np.arange(len(Y[i])),np.argmax(Y[i],axis=1)]

            total += np.sum(np.log(output_probabilities_of_correct_classes) * yz)

        N = np.sum((len(y_i) for y_i in Y))
        loss = (-1. * total) / N
        return loss




    def test_forward_with_reddit_comments():
        vocab_size = 8000
        x_train,y_train = DataPrep.train_for_reddit_comments("../data/reddit-comments-2015-08.csv",vocab_size)

        x_train = x_train[:1]
        y_train_tmp = y_train[:1]
        y_train = []

        for i in np.arange(len(x_train)):
            for k in np.arange(len(x_train[i])):
                x = np.zeros(vocab_size)
                x[x_train[i][k]] = 1
                x_train[i][k] = x

            y = np.zeros((len(y_train_tmp[i]),vocab_size))
            for k in np.arange(len(y_train_tmp[i])):
                y[k][y_train_tmp[i]] = 1

            y_train.append(y)



        rnn = RNN(100,8000,8000)


        O = []
        for i in range(len(x_train)):
            x = x_train[i]

            O.append(rnn.step(x))
            print(i)


        print("Expected Loss for random predictions: %f" % np.log(vocab_size))
        print("Actual loss: %f" % rnn.calculate_loss_OneCorrectClass(x_train,y_train,O))









if __name__ == '__main__':

    RNN.test_forward_with_reddit_comments()