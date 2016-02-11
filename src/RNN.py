import operator

import numpy as np
import scipy as sp

import sys
sys.path.append('../../')

from Util.util.data.DataPrep import *
from Util.util.math.MathUtil import *

class RNN(object):


    def __init__(self, hidden_layer_dim, input_dim, output_dim,bptt_truncate):
        self.hidden_layer_dim = hidden_layer_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bptt_truncate = bptt_truncate
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

    def learning_step(self, x,y, learning_rate):
        o = self.step(x)
        dLdU, dLdV, dLdW = self.calculate_gradients(x,y,o)

        self.U += learning_rate * dLdU
        self.V += learning_rate * dLdV
        self.W += learning_rate * dLdW


    def train(self,x_train,y_train,learning_rate,nepoch,evaluation_step):
        lossZ = []
        seen_examples = 0
        for epoch in range(nepoch):
            if epoch % evaluation_step == 0:
                O = []
                for i in range(len(x_train)):
                    O.append(self.step(x_train[i]))

                loss = self.calculate_loss(x_train,y_train,O)
                lossZ.append(loss)
                print(" Loss after %d examples seen is %f" %(seen_examples, loss))

                if (len(lossZ) > 1) and lossZ[-1] >= lossZ[-2]:
                    learning_rate = learning_rate * 0.5
                    print("Learning Rate: %f: " %learning_rate)



            for i in range(len(x_train)):
                self.learning_step(x_train[i],y_train[i],learning_rate)
                O.append(self.step(x_train[i]))
                seen_examples += 1








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

    def calculate_total_loss(self,X,Y,O):
        total = 0
        for i in np.arange(len(Y)):
            # X[i] ---> Y[i]
            # X[i] ----> O[i]
            e_sum = 0
            for j in np.arange(len(Y[i])):
                e_sum += np.log(O[i][j]).dot(Y[i][j])



            total += e_sum


        return -1 * total

    def calculate_loss(self,X,Y,O):
        total = 0
        for i in np.arange(len(Y)):
            # X[i] ---> Y[i]
            # X[i] ----> O[i]
            # output_probabilities_of_correct_classes = O[i][np.arange(len(Y[i])),np.argmax(Y[i],axis=1)]
            #yz = Y[i][np.arange(len(Y[i])),np.argmax(Y[i],axis=1)]

            e_sum = 0
            for j in np.arange(len(Y[i])):
                epsilon = 1e-15
                O[i][j] = sp.maximum(epsilon, O[i][j])
                #O[i][j] = sp.minimum(1-epsilon, O[i][j])

                e_sum += np.log(O[i][j]).dot(Y[i][j])

            total += e_sum

        N = np.sum([len(y_i) for y_i in Y])
        loss = (-1. * total) / N
        return loss


    def calculate_gradients(self,x,y,o):
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        y = np.array(y)
        o = np.array(o)
        T = len(y)
        delta_o = y - o

        for t  in np.arange(T):
            dLdV += np.outer(delta_o[t],self.S[t].T)
            delta_h = self.V.T.dot(delta_o[t]) * (1 - self.S[t] ** 2)

            for bptt_step in np.arange(max(0,t-self.bptt_truncate),t+1):
                dLdW += np.outer(delta_h.T,self.S[bptt_step - 1])
                dLdU += np.outer(delta_h.T,x[bptt_step])

                delta_h = self.W.T.dot(delta_h) * (1 - self.S[bptt_step-1] ** 2)

        return [dLdU, dLdV, dLdW]


    def gradient_check(self,x,y,o,h=0.001, error_threshold = 0.01):
        o = self.step(x)
        bptt_gradients = self.calculate_gradients(x,y,o)

        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                o = self.step(x)

                gradplus = self.calculate_total_loss([x],[y],[o])
                parameter[ix] = original_value - h
                o = self.step(x)
                gradminus = self.calculate_total_loss([x],[y],[o])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = 0.0
                if (np.abs(backprop_gradient) + np.abs(estimated_gradient)) > 0.:
                    relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                    print("+h Loss: %f" % gradplus)
                    print("-h Loss: %f" % gradminus)
                    print("Estimated_gradient: %f" % estimated_gradient)
                    print("Backpropagation gradient: %f" % backprop_gradient)
                    print("Relative Error: %f" % relative_error)
                    return
                it.iternext()
            print("Gradient check for parameter %s passed." % (pname))



    def test_forward_with_reddit_comments():
        vocab_size = 100
        np.random.seed(10)
        x_train,y_train = DataPrep.train_for_reddit_comments("../data/reddit-comments-2015-08.csv",vocab_size)
            #([[0,1,2,3]], [[1,2,3,4]])


        x_train = x_train[:100]
        y_train_tmp = y_train[:100]
        y_train = []


        for i in np.arange(len(x_train)):
            for k in np.arange(len(x_train[i])):
                x = np.zeros(vocab_size)
                x[x_train[i][k]] = 1
                x_train[i][k] = x

            y = np.zeros((len(y_train_tmp[i]),vocab_size))
            for k in np.arange(len(y_train_tmp[i])):
                y[k][y_train_tmp[i][k]] = 1

            y_train.append(y)



        rnn = RNN(100,vocab_size,vocab_size,4)



        O = []
        for i in range(len(x_train)):
            x = x_train[i]

            O.append(rnn.step(x))
            print(i)


        print("Expected Loss for random predictions: %f" % np.log(vocab_size))
        print("Actual loss: %f" % rnn.calculate_loss(x_train,y_train,O))

        #rnn.gradient_check(x_train[0],y_train[0],O[0])
        losses = rnn.train(x_train,y_train,0.01, 10, 1)









if __name__ == '__main__':

    RNN.test_forward_with_reddit_comments()