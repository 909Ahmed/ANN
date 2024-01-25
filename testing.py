import numpy as np

activations = [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]]

mean = np.mean(activations[-1], axis=0)
std = np.var(activations[-1], axis=0)
stdp = np.sqrt(np.add(std, 0.00000001))
activations[-1] = np.divide(np.subtract(activations[-1], mean), stdp)

print(mean)
print(std)
print(stdp)
print(activations[-1])
# print(mean, std, stdp, activations[-1], end='\n')