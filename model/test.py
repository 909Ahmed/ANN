import numpy as np

array1 = [1, 2, 3, 4, 5, 6, 7]
array2 = [2, 3, 4, 5, 6]

matrix = np.outer(array1, array2)

print(matrix)