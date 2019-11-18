import numpy as np
import time
import sys
import os

from read_format import InputSig,get_input_X_y
from train_sig import LearnSig


results_dir='results'

start_time=time.time()

data=str(sys.argv[1])
embedding=str(sys.argv[2])
algo=str(sys.argv[3])
order=int(sys.argv[4])

if len(sys.argv)==6:
	ll=int(sys.argv[5])
else:
	ll=None

if data=='motion_sense':
	n_test_samples=30
	n_valid_samples=30
	n_train_samples=300
elif data=='urban_sound':
	n_test_samples=500
	n_valid_samples=500
	n_train_samples=4435
elif data=='quick_draw':
	n_train_samples=200*340
	n_valid_samples=20*340
	n_test_samples=20*340


start_row=0
n_processes=32


# Load input data 
inputSig=InputSig(data,embedding,order,ll=ll)

train_X,train_y=get_input_X_y(
	inputSig,n_train_samples,start_row,n_processes=n_processes)
max_train_X=np.amax(np.absolute(train_X),axis=0)
train_X=train_X/max_train_X

valid_X,valid_y=get_input_X_y(
	inputSig,n_valid_samples,start_row+n_train_samples,n_processes=n_processes)
valid_X=valid_X/max_train_X

test_X,test_y=get_input_X_y(
	inputSig,n_test_samples,start_row+n_train_samples+n_valid_samples,
	n_processes=n_processes)
valid_X=valid_X/max_train_X
test_X=test_X/max_train_X

# Fit and evaluate algorithm

learnSig=LearnSig(algo,inputSig,n_processes=n_processes)

# Train with custom parameters, for example for Urban Sound best random forest
custom_params={'n_estimators':460,'max_depth':30,'max_features':500}
learnSig.train(
	train_X,train_y,valid_X=valid_X,valid_y=valid_y,params=custom_params)

#learnSig.train(train_X,train_y,valid_X=valid_X,valid_y=valid_y)
test_results=learnSig.evaluate(test_X,test_y,metrics=['accuracy','f1_score'])
print(test_results)

# Write the results in a separate text file.
results_path=os.path.join(results_dir,learnSig.model_name+".txt")
file = open(results_path,"w")

file.write("Dataset: %s \n" % (data))
file.write("Algorithm: %s \n" % (algo))
file.write("Embedding: %s \n" %(embedding))
file.write("Signature order: %i \n" % (order)) 

file.write("Number of training samples: %i \n" % (n_train_samples))
file.write("Number of validation samples: %i \n" % (n_valid_samples))
file.write("Number of test samples: %i \n" % (n_test_samples))

file.write("Parameters: %s \n" % (learnSig.params))
file.write("First row of training data: %i \n" % (start_row))
file.write("Size of input matrix: %s \n" % (np.shape(train_X),))
file.write("Lag %s \n" % (ll,))

if algo=='neural_network':
	learnSig.model.summary(print_fn=lambda x: file.write(x + '\n'))

file.write("Time %s \n" % (time.time() - start_time))
file.write("Test results: %s" % (test_results))
file.close()






