"""
@author: Adaickalavan Meiyappan
"""

import os
import struct
import theano
from keras.utils import np_utils
import numpy as np
import pdb

#Import the MNIST database
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte' % kind)
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
        
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels
    
path = ''    
X_train, y_train = load_mnist(path, kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist(path, kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

#Cast the MNIST database into 32bit array
theano.config.floatX = 'float32'
X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)

#Convert the class labels into one hot format
print('First 3 labels: ', y_train[:3])
y_train_ohe = np_utils.to_categorical(y_train)
print('\nFirst 3 labels (one-hot):\n', y_train_ohe[:3])

#Implement multi layer perceptron network with Keras library
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

#Set the random number generator
np.random.seed(1)
#Implement a feedforward neural network
model = Sequential()
#Add first fully connected hidden layer using tanh as activation function
model.add(Dense(input_dim=X_train.shape[1], output_dim=50, 
                init='uniform', activation='tanh'))
#Add second fully connected hidden layer using tanh as activation function
model.add(Dense(input_dim=50, output_dim=50, init='uniform', 
                activation='tanh'))
#Add output fully connected layer using softmax as activation function
model.add(Dense(input_dim=50, output_dim=y_train_ohe.shape[1],
                init='uniform', activation='softmax'))
#Define the optimization algorithm to be stochastic gradient descent 
#lr: float >= 0. Learning rate.
#momentum: float >= 0. Parameter updates momentum.
#decay: float >= 0. Learning rate decay over each update.
sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
#Compile the model
#Define the cost function as ‘categorical_crossentropy‘ for multi-class logarithmic loss (logloss)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Train the model using stochatis gradient minibatch of 300 training samples over 50 epochs
#Follow optimization of cost function by setting verbose=1
#Validation plit will reserve 10% of training data for validation, to check for overfitting during training
model.fit(X_train, y_train_ohe, nb_epoch=50, batch_size=300, verbose=1, 
          validation_split=0.1)

#Predict the class labels and print model accuracy on training set
y_train_pred = model.predict_classes(X_train, verbose=0)
print('First 3 predictions: ', y_train_pred[:3])
train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (train_acc * 100))

#Predict the class labels and print model accuracy on test set
y_test_pred = model.predict_classes(X_test, verbose=0)
test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Test accuracy: %.2f%%' % (test_acc * 100))
