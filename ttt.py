import numpy as np

zeroes = 10
ones = 5

mult = np.array(np.concatenate((np.zeros(zeroes), np.ones(ones)), axis=0))
print(mult)
np.random.shuffle(mult)
print(mult)