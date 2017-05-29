"""
	Polynoimal Regression
	=====================
	Features:		x[:,0], x[:,0]**2
	Target value:	Y
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("./data_poly.csv", dtype=float).as_matrix()
X = np.hstack([  np.ones((data.shape[0],1)),  data[:,0].reshape((99,1)),  (data[:,0]**2).reshape((99,1))  ])
Y = data[:,-1]

# W = np.linalg.inv( X.T.dot(X) ).dot( X.T.dot(Y) )
W = np.linalg.solve( X.T.dot(X), X.T.dot(Y) )
Y_hat = X.dot(W)

plt.scatter( X[:,1], Y, c='g' )
X_sorted = sorted(X[:,1])
Y_sorted = sorted(Y_hat)
plt.plot( X_sorted, Y_sorted, 'r' )
plt.show()


# Calculate R^2 measure
ss_res = np.power( Y-Y_hat, 2 ).sum()
ss_tot = np.power( Y-np.mean(Y), 2 ).sum()
print("R^2 measure: {}".format(1-ss_res/ss_tot))
