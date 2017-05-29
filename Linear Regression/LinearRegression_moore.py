"""
	Linear regression to test Moore's law
	Moore's law:
	===========
	The number of transistors per square inch on integrated circuits had doubled every year since their invention.
	That is y(n) = 2*y(n-1) = 2^n, which is an exponential curve
	Taking log on both sides,
	log( y(n) )   =    log(n) + const
	Here, Y = log( y(n) )
	      X = n
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re

X = []
Y = []

non_decimal = re.compile(r"[^\d]+")

for line in open("./moore.csv"):
	line = line.split("\t")
	X.append( non_decimal.sub("", line[2].split('[')[0]) )
	Y.append( non_decimal.sub("", line[1].split('[')[0]) )

X = np.log( np.array(X, dtype=float) )
Y = np.log( np.array(Y, dtype=float) )

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
