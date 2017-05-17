import numpy as np
from matplotlib import pyplot as plt

nClasses = 3
nInpDim = 2
hiddenLayerSize = 4
nSamplesPerClass = 500

def classification_rate(Y, P):
	return (Y==P).tolist().count(True) * 1.0 / len(Y)


def generateData():
	X = np.array([])
	Y = np.array([])

	centroid = [ (-2,3), (2,2), (0,-1) ]

	for i in range(nClasses):
		x = np.random.randn(nSamplesPerClass, nInpDim) + np.array([ centroid[i][0], centroid[i][1] ])
		X = x if len(X)==0 else np.vstack( [X,x] ) 

		y = np.zeros( (nSamplesPerClass,nClasses), dtype=np.int )
		y[:,i] = 1
		Y = y if len(Y)==0 else np.vstack( [Y,y] )

	return X, Y


def softmax( x ):
	exp_x = np.exp(x)
	return exp_x / exp_x.sum(axis=1, keepdims=True)


def sigmoid( x ):
	return 1 / ( 1 + np.exp(-x) )


def gradient_sigmoid(x):
	return x * ( 1-x )


def main():
	X, T = generateData()
	t = np.array([0]*nSamplesPerClass+[1]*nSamplesPerClass+[2]*nSamplesPerClass)
	plt.scatter( X[:,0], X[:,1], c=t, alpha=0.5, s=100 )
	plt.show()

	# Add bias unit to X
	X = np.hstack([ np.ones((X.shape[0],1), dtype=np.float), X ])	

	w_ji = np.random.randn( X.shape[1], hiddenLayerSize + 1 )	# +1 for bias in hidden layer
	w_kj = np.random.randn( hiddenLayerSize + 1, nClasses )		# +1 for bias in hidden layer

	learningRate = 10e-7
	costs = []

	for i in range(50000):
		net_ji = np.dot( X, w_ji )
		Y = sigmoid( net_ji )
		net_kj = np.dot( Y, w_kj )
		Z = softmax( net_kj )
		delta_w_kj = learningRate * Y.T.dot(T-Z)
		delta_w_ji = learningRate * X.T.dot( gradient_sigmoid(Y) * np.dot(T-Z, w_kj.T) )
		w_kj += delta_w_kj
		w_ji += delta_w_ji

		z = np.argmax( Z, axis=1 )
		cost = np.power(t-z ,2).sum()

		if i%100==0:
			costs.append( cost )
			cl_rate = classification_rate(t,z)
			print( "{}. Cost:{}  classification_rate:{}".format(i, cost,cl_rate) )
		if cost <= 10e-7:
			break

	plt.plot(costs)
	plt.show()


if __name__ == '__main__':
	main()
