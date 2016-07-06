import pandas as pd
import numpy as np
import tensorflow as tf
from random import shuffle

class DataSet1(object):
   def __init__(self):
	'''Data set object to manage Galaxy Data and batch approach'''

	#Load Galaxies IDs
	self._list=list(np.genfromtxt('GalaxyID.txt',dtype=int))[:6000]

	np.random.shuffle(self._list)

	self._num_examples=len(self._list)

	#Load dataframe with labels
	self._labels=pd.read_csv('labels.csv',index_col=0)
	self._labels=self._labels.loc[:188668,['Class1.1', 'Class1.2', 'Class1.3']]

	self._num_classes=self._labels.columns.shape[0]

	#read the numer of pixels
	img=pd.read_csv('features/'+str(self._list[0])+'.csv',header=None)
	
	self._num_pixels=img.shape[0]
    	self._epochs_completed = 0
    	self._index_in_epoch = 0


   #-------------------------------------------------------------

   @property
   def list(self):
	return self._list

   @property
   def labels(self):
	return self._labels

   @property
   def num_examples(self):
	return self._num_examples

   @property
   def num_classes(self):
	return self._num_classes

   @property
   def num_pixels(self):
	return self._num_pixels

   @property
   def epochs_completed(self):
	return self._epochs_completed


   #--------------------------------------------------------------

   def labels_batch(self, batch_list):
	return self._labels.loc[batch_list]



   def features_batch(self, batch_list):
	features=[]
	for identity in batch_list:
	   img=np.asarray(pd.read_csv('features/'+str(self._list[0])+'.csv',header=None))
	   features.append(img)
	return np.asarray(features)



   def next_batch(self, batch_size):
	"""Return the next `batch_size` examples from this data set."""
	start = self._index_in_epoch
	self._index_in_epoch += batch_size
	if self._index_in_epoch > self._num_examples:
	   # Finished epoch
	   self._epochs_completed += 1
	   # Shuffle the data
	   np.random.shuffle(self._list)
	   # Start next epoch
	   start = 0
	   self._index_in_epoch = batch_size
	   assert batch_size <= self._num_examples
	end = self._index_in_epoch
	#functions reading labels and features
	labels = np.asarray(self.labels_batch(self._list[start:end]))
	features = self.features_batch(self._list[start:end])
            
	return  features, labels

####################################################################

def load_data():
   return DataSet()


####################################################################




class DataSet(object):
   def __init__(self):
	'''Data set object to manage Galaxy Data and batch approach'''

	#Load Galaxies IDs
	self._list=list(np.genfromtxt('GalaxyID.txt',dtype=int))

	

	#Load dataframe with labels
	ds=pd.read_csv('labels.csv',index_col=0)
	ds=ds.loc[:,['Class1.1', 'Class1.2', 'Class1.3']]

	#self._labels=pd.concat([self._labels[self._labels['Class1.1']>0.9],self._labels[ self._labels['Class1.1']<0.1]])
	self._labels=pd.concat([ds[ds['Class1.1']>0.95],ds[ds['Class1.1']<0.05]])
	#self._labels=ds


	self._list=self._labels.index
	self._num_examples=len(self._list)

	self._num_classes=self._labels.columns.shape[0]
	self._labels=np.asarray(self._labels)

	#read the numer of pixels
	img=pd.read_csv('features/'+str(self._list[0])+'.csv',header=None)
	
	self._num_pixels=img.shape[0]

	#load images
	self._features=[]
	for identity in self._list:
	   img=np.asarray(pd.read_csv('features/'+str(identity)+'.csv',header=None))
	   self._features.append(img)

	self._features=np.asarray(self._features)
    	self._epochs_completed = 0
    	self._index_in_epoch = 0


   #-------------------------------------------------------------

   @property
   def list(self):
	return self._list

   @property
   def labels(self):
	return self._labels

   @property
   def features(self):
	return self._features

   @property
   def num_examples(self):
	return self._num_examples

   @property
   def num_classes(self):
	return self._num_classes

   @property
   def num_pixels(self):
	return self._num_pixels

   @property
   def epochs_completed(self):
	return self._epochs_completed


#-----------------------------------------------------------
   def next_batch(self, batch_size):
	"""Return the next `batch_size` examples from this data set.
	with 'prev'=#minimum fraud examples/batch_size"""
	start = self._index_in_epoch
	self._index_in_epoch += batch_size
	if self._index_in_epoch > self._num_examples:
	   # Finished epoch
     	   self._epochs_completed += 1
	   # Shuffle the data
	   perm = np.arange(self._num_examples)
      	   np.random.shuffle(perm)
      	   self._features = self._features[perm,:,:]
      	   self._labels = self._labels[perm]
	   # Start next epoch
	   start = 0
	   self._index_in_epoch = batch_size
	   assert batch_size <= self._num_examples
	end = self._index_in_epoch
	
	return np.asarray(self._features[start:end]), np.asarray(self._labels[start:end])

