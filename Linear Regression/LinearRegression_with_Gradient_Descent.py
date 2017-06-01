"""
	Linear Regression with Gradient Descent
"""

import matplotlib.pyplot as plt
import numpy as np

N = 10		# number of samples
D = 2 + 1	# dimensionality of data + 1(for bias)

# Data
X = np.zeros((N,D))
X[:,0] = 1
X[:5,1] = 1
X[5:,2] = 1
Y = np.array( [0]*5 + [1]*5 )

W = np.random.randn(D) / np.sqrt(D)  # Gaussian distributed initial weights with mean: 0 & variance: D

etha = 0.001 # Learning rate
costs = []

for i in range(600):
    delta = (X.dot(W)-Y)
    W -= etha * X.T.dot(delta)
    costs.append( delta.dot(delta)/N )

print( "Cost: {}".format(costs[-1]) )
plt.plot(costs)
plt.show()
