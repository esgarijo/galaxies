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
def covnet(ds,learning_rate=0.1, steps=500, batch_size=2**7,N1=32,N2=32,L=0.05,drop_prob=0.5,momentum=0.1):
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
	ds=GalaxyZoo()

	covnet(ds, learning_rate=0.005,N1=10,N2=50, drop_prob=0.15,L=0.003)
	'''

	#---------------------------------------------
	#	DEFINING THE VARIABLES
	#---------------------------------------------

	#Features
	x=tf.placeholder(tf.float32,[None,ds.train,num_pixels, ds.train,num_pixels])
	#Labels
	y_=tf.placeholder(tf.float32,[None,ds.train,num_classes])


	# Reshape input picture
    	x_image = tf.reshape(x, shape=[-1, ds.train,num_pixels, ds.train,num_pixels, 1])

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
		

	W_fc2 = weight_variable([N2, ds.train,num_classes],0.05)
	b_fc2 = bias_variable([ds.train,num_classes])

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
	   x_batch,y_batch=ds.train,next_batch(batch_size)
	   sess.run(train_step,feed_dict={x:x_batch, y_: y_batch,keep_prob: drop_prob})
	   if (i*batch_size % ds.train,num_examples)<batch_size-1:
	      error=sess.run(RMSE,feed_dict={x:x_batch, y_:y_batch, keep_prob: 1})
	      print 'Epoch: ', epoch, '  Loss: ', error
	      epoch+=1

	return sess.run(RMSE,feed_dict={x:x_batch, y_: y_batch, keep_prob: 1})



####################################################################
#
#	COVNET
#	WITH EARLY STOPPING
#
###################################################################




def es_covnet(ds,learning_rate=0.1, steps=500, batch_size=2**7,N1=32,N2=32,L=0.05,drop_prob=0.5,momentum=0.1,p=4):
	''' Engine runing a convultional network on dataset train. Uses Stochastic Gradient Descent 
	Optimizer with momentum to minimize the mean squared error. Architecture:3x3 max_pooling covolutional hidden 
	layer and a fully connected hidden layer. Activation functions:ReLU. Output layer:softmax units.
	 Tikhonov regularization and dropout in the fully connected layer. Early stopping is used.
	PARAMETERS:
	------------------------------
	train : the data set (Loadad using DataSet object from tools.py) to be analyzed.
	learning_rate : learning rate of the SGD algorithm
	batch_size : number of examples over which the net is trained each time
	N1 : width of the convolutional layer
	N2 : width of the hidden fully connected layer
	L : weight of the Tikhonov (L2) regularization
	drop_prob : (Misleading name) probability to keep the output of a neuron of the fully connected layer
	momentum: Parameter of the TensorFlow implementation, I guess it is the "mass". 

	steps : number of actualizations between evaluations on val set
	p : patience: number of times we observe worsening  in val set before giving up


	EXAMPLE:
	from tools import DataSet
	ds=GalaxyZoo()

	covnet(ds, learning_rate=0.005,N1=10,N2=50, drop_prob=0.15,L=0.003)
	'''

	#---------------------------------------------
	#	DEFINING THE VARIABLES
	#---------------------------------------------

	#Features
	x=tf.placeholder(tf.float32,[None,ds.train.num_pixels, ds.train.num_pixels])
	#Labels
	y_=tf.placeholder(tf.float32,[None,ds.train.num_classes])


	# Reshape input picture
    	x_image = tf.reshape(x, shape=[-1, ds.train.num_pixels, ds.train.num_pixels, 1])

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
		

	W_fc2 = weight_variable([N2, ds.train.num_classes],0.05)
	b_fc2 = bias_variable([ds.train.num_classes])

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
	i=0
	j=0
	v=10 #(RMSE can't be higher than 1, this value is infty)


	while j<p:#if we still have patience

	   for k in range(steps):
		x_batch,y_batch,epoch=ds.train.next_batch(batch_size,True)
	   	sess.run(train_step,feed_dict={x:x_batch, y_: y_batch,keep_prob: drop_prob})
		if epoch != None:
	      	   error=sess.run(RMSE,feed_dict={x:ds.train.features, y_:ds.train.labels, keep_prob: 1})
	      	   print 'Epoch: ', epoch, '  Loss: ', error

	   i+=steps
	   v_= sess.run(RMSE,feed_dict={x:ds.val.features, y_:ds.val.labels, keep_prob: 1})

	   if v_<v:#if the validation error decreases
		j=0
		v=v_
		i_=i

	   else: #if the validation error increases
		j+=1
	return v

