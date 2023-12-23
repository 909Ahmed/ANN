from model import Model
from layers import Layer
from Input import Input
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, y_test) = mnist.load_data()

a = Input(784)
b = Layer(64, 'sigmoid', a)
y = Layer(10, 'sigmoid', b)

model = Model (a, y)

X_train = np.array(X_train)
X_train = X_train.reshape (-1, 784)

# print (X_train[0], Y_train)
X_train = X_train / 255
model.fit(X_train, Y_train, 32)