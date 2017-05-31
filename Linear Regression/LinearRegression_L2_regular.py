"""
	Linear Regression with L2 regularizations 
	i.e., Ridge Regression
"""

import matplotlib.pyplot as plt
import numpy as np

N = 100
X = np.linspace(0, 25, N).reshape((N,1))
Y = X*0.5 +  np.random.randn(N).reshape((N,1))

# Adding outliers
Y[70] += 30
Y[85] += 25

# Adding bias
X = np.hstack([ np.ones((N,1)), X ])

plt.scatter( X[:,1], Y, c='y' )

######
##	Linear Regression with NO L2 regularizations 
######
W = np.linalg.solve( X.T.dot(X), X.T.dot(Y) )
Y_hat = X.dot(W)
plt.plot( X[:,1], Y_hat, label="Max. Liklihood" )

# Calculate R^2 measure
ss_res = np.power( Y-Y_hat, 2 ).sum()
ss_tot = np.power( Y-np.mean(Y), 2 ).sum()
print("R^2 measure for Max. Liklihood: {}".format(1-ss_res/ss_tot))


######
##	Linear Regression with L2 regularizations 
######
print("Excluding the utliers:")
L2_lambdas = [ 1000, 2000, 3000 ]
for L2_lambda in L2_lambdas:
	W = np.linalg.solve( X.T.dot(X) + L2_lambda*np.identity(X.shape[1]), X.T.dot(Y) )
	Y_hat = X.dot(W)
	plt.plot( X[:,1], Y_hat, label="L2={}".format(L2_lambda) )

	# Calculate R^2 measure
	Y_tmp = np.vstack([Y[:70], Y[71:85], Y[86:]])
	Y_hat_tmp = np.vstack([Y_hat[:70], Y_hat[71:85], Y_hat[86:]])
	ss_res = np.power( Y_tmp-Y_hat_tmp, 2 ).sum()
	ss_tot = np.power( Y_tmp-np.mean(Y), 2 ).sum()
	print("R^2 measure for L2-{}: {}".format(L2_lambda, 1-ss_res/ss_tot))

plt.legend()
plt.show()
