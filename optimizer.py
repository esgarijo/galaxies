import random #mersenne twister
import numpy as np
import sys
sys.path.append('./')
from tools import DataSet
from engines import es_covnet

trials=[]
N_trials=10

def generate_pair_log(a_1,b_1,a_2,b_2):
	return 10**random.uniform(np.log10(a_1),np.log10(b_1)),10**random.uniform(np.log10(a_2),np.log10(b_2))

def get_pair_u(a_1,b_1,a_2,b_2):
	return random.uniform(a_1,b_1),random.uniform(a_2,b_2)

def generate_parameters(alpha_1,alpha_2,l_1,l_2,d_1,d_2):
	alpha=10**random.uniform(np.log10(alpha_1),np.log10(alpha_2))
	l=10**random.uniform(np.log10(l_1),np.log10(l_2))
	d=random.uniform(d_1,d_2)
	return alpha,l,d


#-------------------------------------------------------------------------

#Load data
ds=DataSet()

Header=['validation_score','train_score','learning_rate','L2_weight','keep_prob']
#log search for optimum learning rate and L2 parameter:
for i in range(N_trials):
	alpha, l, d = generate_parameters(0.0001,0.1,  0.001,0.5,  0.1,0.99)
	print 'Trial', i ,' ; ', alpha, l, d

	v,e=es_covnet(ds, steps=20, p=3, batch_size=64, N1=10,N2=100,drop_prob=d,momentum=0.1, learning_rate=alpha, L=l)
	trials.append([v,e,alpha,l,d])

	print '--------\n'

	if i==0:
		best=v
	elif v<best:
		best=v

np.savetxt('trials.csv',trials,header='validation_score, train_score, learning_rate, L2_weight, keep_prob',delimiter=',')

print 'Best: ', best



