
# input files: train_image.csv, train_label.csv, test_image.csv
# output file: test_predictions.csv
# training: 10000 images

# Hyperparameters
NUM_INPUT = 784
NUM_H1 = 256
NUM_H2 = 128
NUM_H3 = 64
NUM_OUTPUT = 10
LEARNING = 0.01
BATCH_SIZE = 30
NUM_EPOCH = 30


import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import time

'''
Neural Network Model:
Input Layer: 784 (every pixel in 28x28 input image)
Hidden Layer1 (h1): NUM_H1 (relu activation function)
Hidden Layer2 (h2): NUM_H2 (relu activation function)
Hidden Layer2 (h3): NUM_H3 (relu activation function)
Output Layer: 10 (softmax function)

crossentropy error function
SGD learning, lr = 0.3
'''
class NN:
    def __init__(self):
        # initialize and save weights using Xavier weight initialization
        self.w1, self.b1 = self._init_weights(NUM_INPUT, NUM_H1)  
        self.w2, self.b2 = self._init_weights(NUM_H1, NUM_H2)
        self.w3, self.b3 = self._init_weights(NUM_H2, NUM_H3)   
        self.w4, self.b4 = self._init_weights(NUM_H3, NUM_OUTPUT)

    def _init_weights(self, num_input, num_output):
        w = np.random.normal(0.0, 0.01, (num_input, num_output))
        b = np.zeros([1,num_output])
        return w,b
    
    # num_samples ->  num_samples x num_output
    def one_hot_encoded(self, labels):
        n = len(labels)
        output = np.zeros((n, NUM_OUTPUT), dtype=float)
        for i in range(n):
            output[i][labels[i]] = 1.0
        return output

    # apply the function to each element in a matrix
    def _sigmoid(self, x):
        return np.piecewise(x,[x > 0],[lambda i: 1. / (1. + np.exp(-i)), lambda i: np.exp(i) / (1. + np.exp(i))],)
    
    # 1d array
    def _d_sigmoid(self, sigma):
        return sigma * (1.0 - sigma)
    
    def _relu(self, x):
        return np.maximum(0.,x)
    
    def _d_relu(self,x,z):
        x[z<0] = 0.0
        return x

    # apply softmax to each instance
    def _softmax(self, x):
        x_max = np.max(x, axis=1, keepdims=True)
        exp = np.exp(x-x_max)
        return exp / np.sum(exp, axis=1, keepdims=True)

    # average cross loss
    def cross_entropy_loss(self, a3, y):
        m = y.shape[0]
        a3 = np.clip(a3, 1e-12, None)
        return -(1.0 / m) * np.sum(y * np.log(a3))

    # forward propagation
    def forward_propagation(self, X):
        z1 = np.matmul(X, self.w1) + self.b1
        a1 = self._relu(z1)

        z2 = np.matmul(a1, self.w2) + self.b2
        a2 = self._relu(z2)

        z3 = np.matmul(a2, self.w3) + self.b3
        a3 = self._relu(z3)
        
        z4 = np.matmul(a3, self.w4) + self.b4
        a4 = self._softmax(z4)

        a = [a1, a2, a3, a4]
        z = [z1, z2, z3, z4]
        return a4, a, z

    # back propagation
    def back_propagation(self, X, Y, a, z):
        m = X.shape[0]

        a1, a2, a3, a4 = a[0], a[1], a[2], a[3]
        z1, z2, z3, z4 = z[0], z[1], z[2], z[3]

        dz4 = (1.0 / m)*(a4 - Y)
        dw4 = np.dot(a3.T, dz4)
        db4 = np.sum(dz4, axis=0, keepdims = True)

        dz3 = np.dot(dz4, self.w4.T) 
        dz3 = self._d_relu(dz3, z3)
        dw3 = np.dot(a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims = True)
        
        dz2 = np.dot(dz3, self.w3.T)
        dz2 = self._d_relu(dz2, z2)
        dw2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims = True)

        dz1 = np.dot(dz2, self.w2.T)
        dz1 = self._d_relu(dz1, z1)
        dw1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims = True)

        dw = [dw1, dw2, dw3, dw4]
        db = [db1, db2, db3, db4]

        return dw, db

    # update weights
    def update(self, dw, db, learning_rate):
        self.w1 = self.w1 - learning_rate * dw[0]
        self.b1 = self.b1 - learning_rate* db[0]
        
        self.w2 = self.w2 - learning_rate * dw[1]
        self.b2 = self.b2 - learning_rate* db[1]
        
        self.w3 = self.w3 - learning_rate * dw[2]
        self.b3 = self.b3 - learning_rate* db[2]
        
        self.w4 = self.w4 - learning_rate * dw[3]
        self.b4 = self.b4 - learning_rate* db[3]
        return

    def train(self, inputs, labels, learning_rate=LEARNING, num_epoch=NUM_EPOCH, batch_size=BATCH_SIZE):
        targets = self.one_hot_encoded(labels)

        # normalize inputs
        # inputs = inputs/np.linalg.norm(inputs, ord=2, axis=1, keepdims=True)

        N = len(inputs)
        indexes = [i for i in range(N)]

        cost_list = []

        for e in range(num_epoch):
            # shuffle training data
            np.random.shuffle(indexes)

            Y_hat_epoch = np.zeros_like(targets)
            predictions = np.zeros_like(labels)
            curr_start = 0
            while curr_start < N:
                curr_end = min(curr_start + BATCH_SIZE, N)
                X = inputs[indexes[curr_start:curr_end]]
                Y = targets[indexes[curr_start:curr_end]]

                # forward
                y_hat, a, z = self.forward_propagation(X)
                Y_hat_epoch[curr_start:curr_end] = y_hat
                predictions[curr_start:curr_end] = self.output_class(y_hat).flatten()

                # back
                dw, db = self.back_propagation(X, Y, a, z)

                # update parameter with average gradients in a mini-batch
                self.update(dw, db, learning_rate)

                curr_start += BATCH_SIZE

            Y_epoch = targets[indexes]
            cost = self.cross_entropy_loss(Y_hat_epoch, Y_epoch)

            accurate = 0.
            for i in range(len(labels)):
                if labels[indexes[i]] == predictions[i]:
                    accurate += 1.
            accuracy = accurate / len(labels)

            print("At {}th epoch, cost is {}, train accuracy is {}".format(e + 1, cost, accuracy))

        return

        # num_output*10 --> num_output

    def output_class(self, nn_output):
        return np.argmax(nn_output, axis=1)

    # given list of inputs output list of predicted classes
    def test(self, X):
        # normalize inputs
        # X = inputs/np.linalg.norm(X, ord=2, axis=1, keepdims=True)
        nn_output, a,z = self.forward_propagation(X)
        return self.output_class(nn_output)

    
# read image
def read_image_file(filename):
    df = pd.read_csv(filename, header = None)
    return df.values
    #return df.to_numpy()


# read labels
def read_label_file(filename):
    df = pd.read_csv(filename, header = None)
    return df.values.flatten()
    #return df.to_numpy().flatten()

# write output, output is an array of classes
def write_output_file(filename, output):
    output_2d = output.reshape((len(output),-1))
    df = pd.DataFrame(output_2d)
    df.to_csv(filename, header = False, index = False)
    return

if __name__ == "__main__":
    start = time.time()

    # get filenames 
    train_input_filename = sys.argv[1]
    train_label_filename = sys.argv[2]
    test_input_filename = sys.argv[3]
    output_filename = "test_predictions.csv"

    inputs = read_image_file(train_input_filename)
    targets = read_label_file(train_label_filename)
    print(targets.shape)
    test_inputs = read_image_file(test_input_filename)
    network = NN()
    network.train(inputs, targets)
    output = network.test(test_inputs)
    write_output_file(output_filename, output)
    runtime = time.time()-start
    print("Total time: " + str(runtime))


