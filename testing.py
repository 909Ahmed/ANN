import numpy as np

a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
b = [[3, 4, 6], [7, 3, 2], [5, 4, 2], [0, 3, 2]]
c = [3, 4, 5]
d = [4, 5, 6]

print(np.multiply(c, d))
print([np.multiply(x, y) for x, y in zip(a, b)])