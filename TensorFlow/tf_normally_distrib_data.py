"""
	NN with learning from auto-generated normally distributed data.

	Configuration of NN:
	====================
	# input Layer: 		2 units + 1 bias
	# Hidden layer-1:	4 units + 1 bias
	# Output layer:		3 units.
"""


import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

nClasses = 3
nInpDim = 2
hiddenLayerSize = 4
nSamplesPerClass = 500


def generateData():
	x1 = np.random.randn(nSamplesPerClass, nInpDim) + np.array([-2,3])
	x2 = np.random.randn(nSamplesPerClass, nInpDim) + np.array([2,2])
	x3 = np.random.randn(nSamplesPerClass, nInpDim) + np.array([0,-1])
	X = np.vstack( [x1,x2,x3] ).astype(np.float32)
	Y = np.array( [0]*nSamplesPerClass + [1]*nSamplesPerClass + [2]*nSamplesPerClass, dtype=np.int )
	Y_indicator_mat = np.zeros( (Y.shape[0],nClasses), dtype=np.int )
	for i in range(Y.shape[0]):
		Y_indicator_mat[i][Y[i]] = 1
	return X, Y, Y_indicator_mat


def init_weight( shape ):
	return tf.Variable( tf.random_normal(shape, stddev=0.01) )


def main():
	X, T, Y_indicator_mat = generateData()
	plt.scatter( X[:,0], X[:,1], c=T, alpha=0.5, s=100 )
	plt.show()

	# Add bias unit to X
	X = np.hstack([ np.ones((X.shape[0],1), dtype=np.float32), X ])	

	tfX = tf.placeholder( dtype=tf.float32, shape=[None,nInpDim+1] ) # +1 for bias in Input layer
	tfT = tf.placeholder( dtype=tf.float32, shape=[None,nClasses] )

	w_ji = init_weight( [X.shape[1], hiddenLayerSize+1] )	# +1 for bias in hidden layer
	w_kj = init_weight( [hiddenLayerSize+1, nClasses] )		# +1 for bias in hidden layer

	learningRate = 0.5
	costs = []

	net_ji = tf.matmul( tfX, w_ji )
	tfY = tf.nn.sigmoid( net_ji )
	net_kj = tf.matmul( tfY, w_kj )
	Z = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=net_kj,labels=tfT) )	# Cost

	train_opr = tf.train.GradientDescentOptimizer(learningRate).minimize(Z)
	predict_opr = tf.argmax( net_kj, axis=1 )

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	nIterations = 1000
	for i in range(nIterations):
		sess.run( train_opr, feed_dict={tfX:X, tfT:Y_indicator_mat} )
		prediction = sess.run( predict_opr, feed_dict={tfX:X, tfT:Y_indicator_mat} )

		if i%100==0:
			print( "{}. classification_rate:{}".format(i, np.mean(T==prediction)) )
	print( "{}. classification_rate:{}".format(nIterations, np.mean(T==prediction)) )


if __name__ == '__main__':
	main()
