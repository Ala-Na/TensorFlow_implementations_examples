## Python imports

import numpy as np
import tensorflow as tf
# from tensorflow import keras
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.activations import relu, linear, sigmoid
from keras.losses import MeanSquaredError, BinaryCrossentropy, SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.activations import sigmoid

## Create a Sequential model

# A Sequential model is appropriate for a plain stack of layers where each layer
# has exactly one input tensor and one output tensor.
# Sequential model reference : https://keras.io/guides/sequential_model/

model = Sequential(
	[
	tf.keras.Input(shape=(2,)), # Optional, specify expected input size
	Dense(3, activation='sigmoid', name = 'layer1'),
	Dense(units=1, activation='sigmoid', name = 'layer2')
	# Dense for each layer with nb of units (which can be precised or not), type
	# of activation ('sigmoid', 'linear'=Non precised, 'relu', 'softmax'),
	# name of layer, input_dim is the number of dimensions of the features.
	# Everything is optional except the nbr of units. If not precised, default
	# activation is linear activation.
	# Other option : kernel_regularizer which can perform regularization for
	# complex model. Example: tf_regularizer=tf.keras.regularizers.l2(0.1) with
	# a regularization factor lambda of 0.1
	],
	name = "model_name" # Optional
)

# Other implementation :

model = Sequential()
model.add(Dense(1, input_dim=1, activation='sigmoid', name='L1'))

## Print summary of a model

model.summary()

# Example output :
# Model: "my_model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 25)                10025
# _________________________________________________________________
# dense_1 (Dense)              (None, 15)                390
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 16
# =================================================================
# Total params: 10,431
# Trainable params: 10,431
# Non-trainable params: 0
# _________________________________________________________________

## Get weights of a specific layer

logistic_layer = model.get_layer('L1')
w,b = logistic_layer.get_weights()

## Set weights of a specific layer

set_w = np.array([[2]])
set_b = np.array([-4.5])
logistic_layer.set_weights([set_w, set_b])

## Obtaining model prediction
# For a random x train example Xt

prediction = model.predict(Xt)

## Normalize data for a set of features data X

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)

## Defining loss function and compile optimization

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
	# There's many loss algorithms.
	# BinaryCrossentropy : For binary classification.
	# SparseCategoricalCrossentropy(from_logits=True) : For multiclass classification.
	# Note that from_logits precise that value is in [-inf, inf] instead of [0, 1].
	# More losses and options : https://keras.io/api/losses/
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
	# learning_rate can also be not explicity written .Adam(0.01)
	# More optimizers and options : https://keras.io/api/optimizers/
)

## Training model through gradient descent
# For x training set Xt and Y training  set Yt and a defined nbr of epochs (cycles).

model.fit(
    Xt,Yt,
    epochs=10,
)
# Note : model.fit statement can also specify the first layer input data shape if
# it's not done in the modelc.compile statement
# While fitting, the display will show which epoch is run (from 1 to
# defined nbr of max epochs) and which batch is executed (as dataset is cut
# in batch for increasing efficiency).


## Softmax regression
# Used when a problem requires a probability
prediction_softmmax = tf.nn.softmax(prediction)

## More tricks

tf.random.set_seed(1234)
# Weights can be randomly assigned at first. Fixing the seed can help having
# reproductible results when convergence is not yet reached.

## Set variables in order to track them later
# Here W = vector of weights associated to features; X = matrix of features;
# b = weight of scalar type
# Example is based on MovieLens dataset for a collaborative recommendation system,
# there's as many weights as there is users and as many features as there is movies
W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')
