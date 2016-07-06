import pandas as pd
import numpy as np
from random import shuffle

class DataSet(object):
   '''General class from which every set inherits its properties
   labels : pandas dataframe containing the labels assigned to this set'''
   def __init__(self,labels):
	'''Structured way to store data to be processed
	labels : pandas dataframe containing the labels assigned to this set'''

	self._labels=np.asarray(labels)

	#create a list of identifiers
	self._list=labels.index

	self._num_examples=len(self._list)
	self._num_classes=labels.columns.shape[0]
	
	#read the number of pixels
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


#-----------------------------------------------------------


class TrainSet(DataSet):
   pass

   @property
   def epochs_completed(self):
	return self._epochs_completed

   def next_batch(self, batch_size,report=False):
	"""Return the next `batch_size` examples from this data set.
	with 'prev'=#minimum fraud examples/batch_size"""
	epoch=None
	start = self._index_in_epoch
	self._index_in_epoch += batch_size
	if self._index_in_epoch > self._num_examples:
	   # Finished epoch
     	   self._epochs_completed += 1
	   epoch=self._epochs_completed
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
	
	if report:
	   return np.asarray(self._features[start:end]), np.asarray(self._labels[start:end]),epoch
	else:
	   return np.asarray(self._features[start:end]), np.asarray(self._labels[start:end])

   def reset(self):
	self._epochs_completed = 0
    	self._index_in_epoch = 0


################################################################



class GalaxyZoo(object):
   def __init__(self, test_size=0.1):
	'''Data set object to manage Galaxy Data and batch approach
	PARAMTETERS:
	-------------------------------------------------------
	test_size : fraction of data to be used in the test set
	val_size is the same than test
	'''
	#Load dataframe with labels
	ds=pd.read_csv('labels.csv',index_col=0)
	ds=ds.loc[:,['Class1.1', 'Class1.2', 'Class1.3']]

	ds=pd.concat([ds[ds['Class1.1']>0.95],ds[ds['Class1.1']<0.05]])

	#sample a test set
	test=ds.sample(frac=test_size)
	self._test=DataSet(test)

	#drop test  set:
	ds=ds.drop(test.index,axis=0)

	#sample a validation set
	val=ds.sample(n=self._test.num_examples)
	self._val=DataSet(val)

	#drop val to get train:
	train=ds.drop(val.index,axis=0)

	#create train set:
	self._train=TrainSet(train)


   @property
   def test(self):
	return self._test

   @property
   def val(self):
	return self._val

   @property
   def train(self):
	return self._train

	