"""
	Linear regression
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("./data_1d.csv")
data = df.as_matrix()
X, Y = data[:,0], data[:,1]

denom = np.mean(X*X) - np.power( np.mean(X), 2 )
a_num = np.mean(X*Y) - np.mean(X)*np.mean(Y)
b_num = np.mean(Y)*np.mean(X*X) - np.dot( np.mean(X), np.mean(X*Y) )

a = a_num / denom
b = b_num / denom

plt.scatter(X,Y)
plt.plot(X,a*X + b, 'r')
plt.show()


# Calculate R^2 measure
ss_res = np.power( Y-(a*X+b), 2 ).sum()
ss_tot = np.power( Y-np.mean(Y), 2 ).sum()
print("R^2 measure: {}".format(1-ss_res/ss_tot))
