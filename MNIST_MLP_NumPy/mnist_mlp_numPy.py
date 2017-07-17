"""
@author: Adaickalavan Meiyappan
"""

import os
import struct
import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.special import expit #expit function is the logistic sigmoid function expit(x)=1/(1+exp(-x))
import sys

#Import MNIST database 
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        #pdb.set_trace()
        
    return images, labels
        
#Load the training and test data
path = ''
X_train, y_train = load_mnist(path, kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist(path, kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

#Check how the ten digits are for a given handwriting
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

#Check how different handwritings are for a given digit 
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.
    n_output : number of unique class labels
    n_features : feature dimension / number of columns in the X matrix
    n_hidden : Number of hidden units.        
    l2 : Lambda value for L2-regularization.
    epochs : Number of passes over the training set.        
    eta : Learning rate
    alpha : Forgetting factor
    decrease_const : to reduce learning rate after every epoch 
    shuffle : to shuffle training data every epoch
    minibatches : Divides training data into batches of size minibatches
    random_state : Set random state for shuffling and initializing the weights
    cost_ : Cross entropy cost 
    """
    def __init__(self, n_output, n_features, n_hidden=30,
                 l2=0.0, epochs=500, eta=0.001,
                 alpha=0.0, decrease_const=0.0, shuffle=True,
                 minibatches=1, random_state=None):

        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _encode_labels(self, y, k):
        """Encode labels into one-hot representation
        y : Target labels 
        onehot : array, shape = (n_labels, n_samples)
        """
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):
        """Initialize weights with small random numbers."""
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden*(self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output*(self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        return w1, w2

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)
        Uses scipy.special.expit to avoid overflow
        error for very small input values z.
        """
        # return 1.0 / (1.0 + np.exp(-z))
        return expit(z)

    def _sigmoid_gradient(self, z):
        """Compute gradient of the logistic function"""
        sg = self._sigmoid(z)
        return sg * (1.0 - sg)

    def _add_bias_unit(self, X, how='column'):
        """Add bias unit (column or row of 1s) to array at index 0"""
        
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new

    def _feedforward(self, X, w1, w2):
        """Compute feedforward step
        X : Input layer with original features.
        w1 : Weight matrix for input layer -> hidden layer.
        w2 : Weight matrix for hidden layer -> output layer.
        a1 : Input values with bias unit.
        z2 : Net input of hidden layer.
        a2 : Activation of hidden layer.
        z3 :  Net input of output layer.
        a3 : Activation of output layer.
        """
        a1 = self._add_bias_unit(X, how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3

    def _L2_reg(self, lambda_, w1, w2):
        """Compute L2-regularization cost"""
        return (lambda_/2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))

    def _get_cost(self, y_enc, output, w1, w2):
        """Compute cost function.
        y_enc : one-hot encoded class labels.
        output : Activation of the output layer (feedforward)
        w1 :  Weight matrix for input layer -> hidden layer.
        w2 : Weight matrix for hidden layer -> output layer.
        cost : Regularized cost.
        """
        term1 = -y_enc * (np.log(output))
        term2 = (1.0 - y_enc) * np.log(1.0 - output)
        cost = np.sum(term1 - term2)
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L2_term
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        """ Compute gradient step using backpropagation.
        a1 : Input values with bias unit.
        a2 : Activation of hidden layer.
        a3 : Activation of output layer.
        z2 : Net input of hidden layer.
        y_enc : one-hot encoded class labels.
        w1 : Weight matrix for input layer -> hidden layer.
        w2 : Weight matrix for hidden layer -> output layer.
        grad1 : Gradient of the weight matrix w1.
        grad2 : Gradient of the weight matrix w2.
        """
        # backpropagation
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)

        # regularize
        grad1[:, 1:] += self.l2 * w1[:, 1:]
        grad2[:, 1:] += self.l2 * w2[:, 1:]

        return grad1, grad2

    def predict(self, X):
        """Predict class labels
        X : Input layer 
        y_pred : Predicted labels
        """
        if len(X.shape) != 2:
            raise AttributeError('X must be a [n_samples, n_features] array.')

        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)
        return y_pred

    def fit(self, X, y, print_progress=False):
        """ Learn weights from training data.
        X : Input layer
        y : Target labels
        """
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):

            # adaptive learning rate
            self.eta /= (1 + self.decrease_const*i)

            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_enc = X_data[idx], y_enc[:, idx]
 
            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:

                # feedforward
                a1, z2, a2, z3, a3 = self._feedforward(X_data[idx], self.w1, self.w2)
                cost = self._get_cost(y_enc=y_enc[:, idx], output=a3, w1=self.w1, w2=self.w2)
                self.cost_.append(cost)

                # compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2, a3=a3, z2=z2, 
                                                  y_enc=y_enc[:, idx],
                                                  w1=self.w1, w2=self.w2)

                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

        return self
   
#Initialize a multi layer perceptron with 784 inputs, 50 hidden units, 10 output units      
nn = NeuralNetMLP(n_output=10, n_features=X_train.shape[1], n_hidden=50,
    l2=0.1, epochs=50, eta=0.001, alpha=0.001, decrease_const=0.00001,
    shuffle=True, minibatches=50, random_state=1)
    
#Train the neural network using 60,000 training samples
nn.fit(X_train, y_train, print_progress=True)

#Visualize convergence in the cost function    
plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 5000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')
plt.tight_layout()
plt.show()

#Evaluate the model's accuracy on training set
y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))

#Evaluate the model's accuracy on test set
y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))