import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D
import numpy as np
import pandas as pd

data = pd.read_csv("./data_2d.csv").as_matrix()
X, Y = data[:,:-1], data[:,-1]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter( X[:,0], X[:,1], Y )
plt.show()

# W = np.linalg.inv( X.T.dot(X) ).dot( X.T.dot(Y) )
W = np.linalg.solve( X.T.dot(X), X.T.dot(Y) )
Y_hat = X.dot(W)


# Calculate R^2 measure
ss_res = np.power( Y-Y_hat, 2 ).sum()
ss_tot = np.power( Y-np.mean(Y), 2 ).sum()
print("R^2 measure: {}".format(1-ss_res/ss_tot))
