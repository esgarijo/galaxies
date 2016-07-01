import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

#-----------------------------------------------------
#   DEFINING VARIABLE TEMPLATE
#-----------------------------------------------------
#variables are initialized to positive small numbers to break symmetry


def weight_variable(shape):

  initial = tf.truncated_normal(shape, stddev=0.1)
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




####################################################
#
#	THE PERCEPTRON CLASSIFIER
#
####################################################


def perceptron(Train,learning_rate=0.5, steps=500, batch_size=2**7):
	
	#---------------------------------------------
	#	DEFINING THE VARIABLES
	#---------------------------------------------

	#Features
	x=tf.placeholder(tf.float32,[None,Train.num_pixels, Train.num_pixels])
	#Labels
	y_=tf.placeholder(tf.float32,[None,Train.num_classes])


	#Weights
	W = weight_variable([Train.num_pixels**2,Train.num_classes])
	#Biases
	b = bias_variable([Train.num_classes])

	#-------------------------------------------
	#	BUILDING THE MODEL
	#-------------------------------------------
	
	#Reshape the input
	z = tf.reshape(x,shape = [-1,Train.num_pixels**2])

	#Obtain  output
	y = tf.nn.softmax(tf.matmul(z,W)  +b)

	#--------------------------------------------------
	#	LOSS FUNCTION
	#--------------------------------------------------

	RMSE = tf.reduce_mean((y_-y)**2)
	loss_function=RMSE+0.8*tf.reduce_mean(W**2)
	train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

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

	lista=[]	
	
	for i in range(steps):
	   x_batch,y_batch=Train.next_batch(batch_size)
	   sess.run(train_step,feed_dict={x:x_batch, y_: y_batch})
	   lista.append(sess.run(RMSE,feed_dict={x:x_batch, y_:y_batch}))

	plt.plot(lista)
	plt.show()

	return sess.run(RMSE,feed_dict={x:x_batch, y_: y_batch})



#################################################################
#
#	MULTILAYER PERCEPTRON
#	1 HIDDEN LAYER
#
#################################################################



def MLP_1hidden(Train,learning_rate=0.5, steps=500, batch_size=2**7,N1=100):

	#---------------------------------------------
	#	DEFINING THE VARIABLES
	#---------------------------------------------

	#Features
	x=tf.placeholder(tf.float32,[None,Train.num_pixels, Train.num_pixels])
	#Labels
	y_=tf.placeholder(tf.float32,[None,Train.num_classes])

	#Reshape the input
	z = tf.reshape(x,shape = [-1,Train.num_pixels**2])

	#---------------------------------------------
	#	HIDDEN LAYER
	#---------------------------------------------
	
	W1 = weight_variable([Train.num_pixels**2,N1])
	b1 = bias_variable([N1])

	#output
	h1 = tf.nn.relu(tf.matmul(z, W1) + b1)

	#--------------------------------------------
	#	OUTPUT LAYER
	#--------------------------------------------

	W2 = weight_variable([N1,Train.num_classes])
	b2 = bias_variable([Train.num_classes])

	#Obtain  output
	y = tf.nn.softmax(tf.matmul(h1,W2)  +b2)

	#--------------------------------------------------
	#	LOSS FUNCTION
	#--------------------------------------------------

	RMSE = tf.reduce_mean((y_-y)**2)
	loss_function=RMSE#+0.04*tf.reduce_mean(W1**2)+0.03*tf.reduce_mean(W2**2)
	train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

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

	lista=[]	
	
	for i in range(steps):
	   x_batch,y_batch=Train.next_batch(batch_size)
	   sess.run(train_step,feed_dict={x:x_batch, y_: y_batch})
	   lista.append(sess.run(RMSE,feed_dict={x:x_batch, y_:y_batch}))

	plt.plot(lista)
	plt.show()

	return sess.run(RMSE,feed_dict={x:x_batch, y_: y_batch})



#################################################################
#
#	MULTILAYER PERCEPTRON
#	2 HIDDEN LAYER
#
#################################################################
def MLP_2hidden(Train,learning_rate=0.5, steps=500, batch_size=2**7,N1=100, N2=10):

	#---------------------------------------------
	#	DEFINING THE VARIABLES
	#---------------------------------------------

	#Features
	x=tf.placeholder(tf.float32,[None,Train.num_pixels, Train.num_pixels])
	#Labels
	y_=tf.placeholder(tf.float32,[None,Train.num_classes])

	#Reshape the input
	z = tf.reshape(x,shape = [-1,Train.num_pixels**2])

	#---------------------------------------------
	#	HIDDEN LAYER 1
	#---------------------------------------------
	
	W1 = weight_variable([Train.num_pixels**2,N1])
	b1 = bias_variable([N1])

	#output
	h1 = tf.nn.relu(tf.matmul(z, W1) + b1)

	#---------------------------------------------
	#	HIDDEN LAYER 2
	#---------------------------------------------
	
	W2 = weight_variable([N1,N2])
	b2 = bias_variable([N2])

	#output
	h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

	#--------------------------------------------
	#	OUTPUT LAYER
	#--------------------------------------------

	W3 = weight_variable([N2,Train.num_classes])
	b3 = bias_variable([Train.num_classes])

	#Obtain  output
	y = tf.nn.softmax(tf.matmul(h2,W3)  +b3)

	#--------------------------------------------------
	#	LOSS FUNCTION
	#--------------------------------------------------

	RMSE = tf.reduce_mean((y_-y)**2)
	loss_function=RMSE#+0.04*tf.reduce_mean(W1**2)+0.03*tf.reduce_mean(W2**2)
	train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

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

	lista=[]	
	
	for i in range(steps):
	   x_batch,y_batch=Train.next_batch(batch_size)
	   sess.run(train_step,feed_dict={x:x_batch, y_: y_batch})
	   lista.append(sess.run(RMSE,feed_dict={x:x_batch, y_:y_batch}))

	plt.plot(lista)
	plt.show()

	return sess.run(RMSE,feed_dict={x:x_batch, y_: y_batch})




####################################################################
#
#	COVNET
#
###################################################################
def covnet(Train,learning_rate=0.1, steps=500, batch_size=2**7,N1=32,N2=32,L=0.05):

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
	W_conv1 = weight_variable([5, 5, 1, N1])
	b_conv1 = bias_variable([N1])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_3x3(h_conv1)

	#--------------------------------------------
	#	FULLY CONECTED LAYER
	#--------------------------------------------

	W_fc1 = weight_variable([23 * 23 * N1, N2])
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
		

	W_fc2 = weight_variable([N2, Train.num_classes])
	b_fc2 = bias_variable([Train.num_classes])

	y= tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc2) + b_fc2)


	#--------------------------------------------------
	#	LOSS FUNCTION
	#--------------------------------------------------

	RMSE = tf.reduce_mean((y_-y)**2)
	loss_function=RMSE+L*(tf.reduce_mean(W_fc1**2)+tf.reduce_mean(W_fc2**2)+tf.reduce_mean(W_conv1**2))
	train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

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

	#lista=[]	
	
	for i in range(steps):
	   x_batch,y_batch=Train.next_batch(batch_size)
	   sess.run(train_step,feed_dict={x:x_batch, y_: y_batch,keep_prob: 0.5})
	   #lista.append(sess.run(RMSE,feed_dict={x:x_batch, y_:y_batch, keep_prob: 1}))

	#plt.plot(lista)
	#plt.show()

	return sess.run(RMSE,feed_dict={x:x_batch, y_: y_batch, keep_prob: 1})

