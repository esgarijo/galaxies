import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

#-----------------------------------------------------
#   DEFINING VARIABLE TEMPLATE
#-----------------------------------------------------
#variables are initialized to positive small numbers to break symmetry


def weight_variable(shape,std=0.1):

  initial = tf.truncated_normal(shape, stddev=std)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#------------------------------------------------------
#	TEMPLATES FOR CONVS AND POOLS	
#------------------------------------------------------
# Create some wrappers for simplicity
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_3x3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='SAME')






####################################################################
#
#	COVNET
#
###################################################################
def covnet(Train,learning_rate=0.1, steps=500, batch_size=2**7,N1=32,N2=32,L=0.05,drop_prob=0.5,momentum=0.1):
	''' Engine runing a convultional network on dataset Train. Uses Stochastic Gradient Descent 
	Optimizer with momentum to minimize the mean squared error. Architecture:3x3 max_pooling covolutional hidden 
	layer and a fully connected hidden layer. Activation functions:ReLU. Output layer:softmax units.
	 Tikhonov regularization and dropout in the fully connected layer.
	PARAMETERS:
	------------------------------
	Train : the data set (Loadad using DataSet object from tools.py) to be analyzed.
	learning_rate : learning rate of the SGD algorithm
	steps : number of actualizations
	batch_size : number of examples over which the net is trained each time
	N1 : width of the convolutional layer
	N2 : width of the hidden fully connected layer
	L : weight of the Tikhonov (L2) regularization
	drop_prob : (Misleading name) probability to keep the output of a neuron of the fully connected layer
	momentum: Parameter of the TensorFlow implementation, I guess it is the "mass". 


	EXAMPLE:
	from tools import DataSet
	ds=DataSet()

	covnet(ds, learning_rate=0.005,N1=10,N2=50, drop_prob=0.15,L=0.003)
	'''

	#---------------------------------------------
	#	DEFINING THE VARIABLES
	#---------------------------------------------

	#Features
	x=tf.placeholder(tf.float32,[None,Train.num_pixels, Train.num_pixels])
	#Labels
	y_=tf.placeholder(tf.float32,[None,Train.num_classes])


	# Reshape input picture
    	x_image = tf.reshape(x, shape=[-1, Train.num_pixels, Train.num_pixels, 1])

	#---------------------------------------------
	#	CONVOLUTIONAL LAYER
	#---------------------------------------------
	W_conv1 = weight_variable([5, 5, 1, N1],0.05)
	b_conv1 = bias_variable([N1])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_3x3(h_conv1)

	#--------------------------------------------
	#	FULLY CONECTED LAYER
	#--------------------------------------------

	W_fc1 = weight_variable([23 * 23 * N1, N2],0.05)
	b_fc1 = bias_variable([N2])

	h_pool1_flat = tf.reshape(h_pool1, [-1, 23 * 23 * N1])

	h_pool2_flat = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

	#--------------------------------------------
	#	DROPOUT
	#--------------------------------------------
	keep_prob = tf.placeholder(tf.float32)
	h_fc2_drop = tf.nn.dropout(h_pool2_flat, keep_prob)


	#--------------------------------------------
	#	FULLY CONECTED OUTPUT LAYER
	#--------------------------------------------
		

	W_fc2 = weight_variable([N2, Train.num_classes],0.05)
	b_fc2 = bias_variable([Train.num_classes])

	y= tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc2) + b_fc2)


	#--------------------------------------------------
	#	LOSS FUNCTION
	#--------------------------------------------------

	RMSE = tf.reduce_mean((y_-y)**2)
	loss_function=RMSE+L*(tf.reduce_mean(W_fc1**2)+tf.reduce_mean(W_fc2**2)+tf.reduce_mean(W_conv1**2))
	train_step=tf.train.MomentumOptimizer(learning_rate,momentum).minimize(loss_function)

	#---------------------------------------------------
	#       INITIALIZE VARIABLES and 
	#       LAUNCH SESSION
	#---------------------------------------------------
	init=tf.initialize_all_variables()

	sess=tf.Session()
	sess.run(init)

	#--------------------------------------------------
	#	TRAINING
	#--------------------------------------------------

	epoch=0
	for i in range(steps):
	   x_batch,y_batch=Train.next_batch(batch_size)
	   sess.run(train_step,feed_dict={x:x_batch, y_: y_batch,keep_prob: drop_prob})
	   if (i*batch_size % Train.num_examples)<batch_size-1:
	      error=sess.run(RMSE,feed_dict={x:x_batch, y_:y_batch, keep_prob: 1})
	      print 'Epoch: ', epoch, '  Loss: ', error
	      epoch+=1

	return sess.run(RMSE,feed_dict={x:x_batch, y_: y_batch, keep_prob: 1})

