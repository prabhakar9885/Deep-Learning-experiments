import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

def get_data():
	df = pd.read_csv("./iris.data")
	data = df.as_matrix()

	X, Y = data[:, :-1], data[:, -1]
	for i in range( X.shape[1] ):
		X[:,i] = ( X[:,i]-X[:,i].mean() ) / X[:,i].std()

	N, D = X.shape
	X2 = np.zeros((N,D+3))
	X2[:, 0:(D-1)] = X[:, 0:(D-1)]

	Z = np.zeros((N,4))
	Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
	X2[:,-4:] = Z

	return X2, Y


def Z_indicator(T):
	n = len(T)
	nClasses = len(set(T))
	t = np.zeros((n,nClasses))
	for i in range(n):
		t[i,T[i]] = 1
	return t


def get_binary_data():
	X,Y = get_data()
	X2 = X[Y<=1]
	Y2 = Y[Y<=1]
	return X2, Y2

def classification_rate(Y, P):
	return (Y==P).tolist().count(True) * 1.0 / len(Y)

def softmax( x ):
	exp_x = np.exp(x)
	return exp_x / exp_x.sum(axis=1, keepdims=True)


def sigmoid( x ):
	return 1 / ( 1 + np.exp(-x) )


def gradient_sigmoid(x):
	return x * ( 1-x )


def name_to_numbers(T):
	s = set(T)
	d = {}
	indx = 0
	for ele in s:
		if ele not in d:
			d[ele] = indx
			indx += 1
	nums = []
	for t in T:
		nums.append( d[t] )
	return np.array(nums)


def main():
	X, T = get_data()
	T = name_to_numbers(T)

	nInpDim = X.shape[1]
	nClasses = len(set(T))
	hiddenLayerSize = 8
	nSamplesPerClass = T.shape[0]

	t = Z_indicator(T)

	# Add bias unit to X
	X = np.hstack([ np.ones((X.shape[0],1), dtype=np.float), X ])	

	w_ji = np.random.randn( X.shape[1], hiddenLayerSize + 1 )	# +1 for bias in hidden layer
	w_kj = np.random.randn( hiddenLayerSize + 1, nClasses )		# +1 for bias in hidden layer

	learningRate = 10e-5
	costs = []
	cl_rates = []

	for i in range(100000):
		net_ji = np.dot( X, w_ji )
		Y = sigmoid( net_ji )
		net_kj = np.dot( Y, w_kj )
		Z = softmax( net_kj )
		delta_w_kj = learningRate * Y.T.dot(t-Z)
		delta_w_ji = learningRate * X.T.dot( gradient_sigmoid(Y) * np.dot(t-Z, w_kj.T) )
		w_kj += delta_w_kj
		w_ji += delta_w_ji

		z = np.argmax( Z, axis=1 )
		cost = np.power(T-z ,2).sum()

		if i%100==0:
			cl_rate = classification_rate(T,z)
			costs.append( cost )
			cl_rates.append( cl_rate )
			print( "{}. Cost:{}  classification_rate:{}".format(i, cost,cl_rate) )
		if cost <= 10e-7:
			break

	plt.plot(costs, label="Costs")
	plt.plot(cl_rates, label="Classification rate")
	plt.show()


if __name__ == '__main__':
	main()
