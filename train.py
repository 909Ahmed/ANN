from cheras.Model.model import Model
from cheras.Layers.Dense import Dense
from cheras.Layers.Input import Input
from cheras.Layers.Dropout import Dropout
import numpy as np
import pandas as pd

train = pd.read_csv('./data/mnist_train.csv')
test = pd.read_csv('./data/mnist_test.csv')

X_train = train.iloc[:,1:]
X_test = test.iloc[:,1:]

Y_train = train.iloc[:,0]
Y_test = test.iloc[:,0]

X_train = np.array(X_train)
X_train = X_train.reshape (-1, 784)
X_train = X_train / 255

X_test = np.array(X_test)
X_test = X_test.reshape (-1, 784)
X_test = X_test / 255

def get_model():

    i = Input(784)
    x = Dense(i, 64, 'sigmoid')
    x = Dense(x, 64, 'sigmoid')
    x = Dense(x, 10, 'sigmoid')

    model = Model (i, x)
    return model

model = get_model()
model.optimize(0.001, 0.9, 0.999)   #only adam
model.fit(X_train, Y_train, X_test, Y_test, 5, 8)
