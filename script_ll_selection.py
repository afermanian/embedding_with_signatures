import numpy as np
import time
import sys
import os
import pandas as pd

from read_format import InputSig,get_input_X_y
from train_sig import LearnSig



results_dir='results'

start_time=time.time()

data=str(sys.argv[1])
algo='neural_network'
ll_list=np.arange(3)

if data=='motion_sense':
	n_test_samples=30
	n_valid_samples=30
	n_train_samples=300
	order_dict={
		'0':np.arange(2)+1,'1':np.arange(2)+1,'2':np.arange(2)+1,
		'3':np.arange(2)+1,'4':np.arange(2)+1,'5':np.arange(2)+1,
		'6':np.arange(2)+1}
	# order_dict={
	# 	'0':np.arange(5)+1,'1':np.arange(4)+1,'2':np.arange(4)+1,
	# 	'3':np.arange(4)+1,'4':np.arange(3)+1,'5':np.arange(3)+1,
	# 	'6':np.arange(3)+1}
elif data=='urban_sound':
	n_test_samples=500
	n_valid_samples=500
	n_train_samples=4435
	order_dict={
		'0':np.arange(14)+1,'1':np.arange(9)+1,'2':np.arange(8)+1,
		'3':np.arange(7)+1,'4':np.arange(6)+1,'5':np.arange(6)+1,
		'6':np.arange(5)+1}
elif data=='quick_draw':
	n_train_samples=200*340
	n_valid_samples=20*340
	n_test_samples=20*340
	order_dict={
		'0':np.arange(10)+1,'1':np.arange(7)+1,'2':np.arange(6)+1,
		'3':np.arange(5)+1,'4':np.arange(5)+1,'5':np.arange(4)+1,
		'6':np.arange(4)+1}


start_row=0
n_processes=32
results_df=pd.DataFrame({
	'accuracy':[],'embedding':[],'algo':[],'order':[],'ll':[],
	'n_features':[]})


# Load input data 
for ll in ll_list:
	if ll==0:
		embedding='time'
	else:
		embedding='lead_lag'
	for order in order_dict[str(ll)]:
		inputSig=InputSig(data,embedding,order,ll=ll)

		train_X,train_y=get_input_X_y(
			inputSig,n_train_samples,start_row,n_processes=n_processes)
		max_train_X=np.amax(np.absolute(train_X),axis=0)
		train_X=train_X/max_train_X

		valid_X,valid_y=get_input_X_y(
			inputSig,n_valid_samples,start_row+n_train_samples,
			n_processes=n_processes)
		valid_X=valid_X/max_train_X

		test_X,test_y=get_input_X_y(
			inputSig,n_test_samples,start_row+n_train_samples+n_valid_samples,
			n_processes=n_processes)
		valid_X=valid_X/max_train_X
		test_X=test_X/max_train_X

		# Fit and evaluate algorithm
		learnSig=LearnSig(algo,inputSig)
		learnSig.train(train_X,train_y,valid_X=valid_X,valid_y=valid_y)
		test_results=learnSig.evaluate(test_X,test_y,metrics=['accuracy'])
		print(test_results)

		results_df=results_df.append(
			{'accuracy':test_results['accuracy'],'embedding':embedding,
			'algo':algo,'order':order,'ll':ll,'n_features':train_X.shape[1]},
			ignore_index=True)
		print(results_df)


		# Write the results in a separate text file.

		all_results_dir=os.path.join(results_dir,'results_all_files')
		if not os.path.exists(all_results_dir):
			os.mkdir(all_results_dir)

		results_path=os.path.join(all_results_dir,learnSig.model_name+".txt")
		file = open(results_path,"w")

		file.write("Dataset: %s \n" % (inputSig.data))
		file.write("Algorithm: %s \n" % (learnSig.algo))
		file.write("Embedding: %s \n" %(inputSig.embedding))
		file.write("Lag %s \n" % (inputSig.ll))
		file.write("Signature order: %i \n" % (inputSig.order)) 

		file.write("Number of training samples: %i \n" % (n_train_samples))
		file.write(
			"Number of validation samples: %i \n" % (n_valid_samples))
		file.write("Number of test samples: %i \n" % (n_test_samples))

		file.write("Parameters: %s \n" % (learnSig.params))
		file.write("First row of training data: %i \n" % (start_row))
		file.write("Size of input matrix: %s \n" % (np.shape(train_X),))

		if algo=='neural_network':
			learnSig.model.summary(print_fn=lambda x: file.write(x + '\n'))

		file.write("Time %s \n" % (time.time() - start_time))
		file.write("Test results: %s" % (test_results))
		file.close()

results_df.to_csv(os.path.join(results_dir,'%s_lag_selection_study.csv'%(data)))

