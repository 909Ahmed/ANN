from model import Model
from layers import Layer
from Input import Input

a = Input(10)
b = Layer(64, 'sigmoid', a)
y = Layer(10, 'sigmoid', b)

model = Model (a, y)

model.fit([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 1, 32)