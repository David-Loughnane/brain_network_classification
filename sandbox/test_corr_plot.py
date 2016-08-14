import numpy as np
from matplotlib import pyplot as plt


A = 2 * np.random.ranf(24336).reshape(156, 156) - 1
for i in range(156):
	A[i][i] = 1
plt.imshow(A,interpolation='nearest')
plt.colorbar()
plt.title("Functional Correlation")
plt.xlabel('Parcels')
plt.ylabel('Parcels')
plt.show()