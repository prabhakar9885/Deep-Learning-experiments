"""
	Neural Net trained on batch data with 
	 + L1 regularization and
	 + Momentum
"""


import gzip, os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

def load_mnist(data_dir=None):
    fd = gzip.open(os.path.join(data_dir,'train-images-idx3-ubyte.gz'))
    loaded = np.fromstring(fd.read(), dtype=np.uint8)
    trX = loaded[16:].reshape((60000, -1))

    fd = gzip.open(os.path.join(data_dir,'train-labels-idx1-ubyte.gz'))
    loaded = np.fromstring(fd.read(), dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    # normalization
    trX = trX / 255

    return trX, trY

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


def relu(x):
	x[x<0] = 0
	return x

def gradient_relu(x):
	x = (x>0) * 1
	return x


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
	X, T = load_mnist("/home/prabhakar/Downloads/")

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

	momentum = 0.9
	delta_w_kj_prev = 0
	delta_w_ji_prev = 0

	reg = 0.01
	delta_w_kj = 0
	delta_w_ji = 0

	batch_size = 6000
	num_of_batches = int(X.shape[0]/batch_size)

	for i in range(100):
		for batch_num in range(num_of_batches):
			X_batch = X[(batch_num)*batch_size:(batch_num+1)*batch_size]
			T_batch = t[(batch_num)*batch_size:(batch_num+1)*batch_size]
			net_ji = np.dot( X, w_ji )
			Y = sigmoid( net_ji )
			net_kj = np.dot( Y, w_kj )
			Z = softmax( net_kj )
			delta_w_kj = learningRate * (Y.T.dot(t-Z) + momentum*delta_w_kj_prev + reg*delta_w_kj)
			delta_w_ji = learningRate * (X.T.dot(gradient_sigmoid(Y)*np.dot(t-Z, w_kj.T)) + momentum*delta_w_ji_prev + reg*delta_w_ji)
			w_kj += delta_w_kj
			w_ji += delta_w_ji
			delta_w_kj_prev = delta_w_kj
			delta_w_ji_prev = delta_w_ji

		z = np.argmax( Z, axis=1 )
		cost = np.power(T-z ,2).sum()

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
