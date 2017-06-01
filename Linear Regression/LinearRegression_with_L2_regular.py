"""
	Linear Regression with Gradient Descent
"""

import matplotlib.pyplot as plt
import numpy as np

N = 50		# number of samples
D = 50	# dimensionality of data + 1(for bias)

# Data
X = np.random.random((N,D))*10 - 5 # Mean= 0, Variance=5
true_W = np.array( [-0.5, 1, 2.5] + [0]*(D-3) )
Y = X.dot(true_W) + np.random.randn(N)*0.5

W = np.random.randn(D) / np.sqrt(D)  # Gaussian distributed initial weights with mean: 0 & variance: D
l1 = 100
etha = 1e-4 # Learning rate
costs = []

for i in range(200):
    delta = X.dot(W) - Y
    W -= etha*( X.T.dot(delta) + l1*np.sign(W) )
    costs.append( delta.dot(delta)/N )

print( "Cost: {}".format(costs[-1]) )
plt.plot(costs)
plt.show()

plt.plot(true_W, label="true W")
plt.plot(W, label="W with Gradient Descent")
plt.legend()
plt.show()
