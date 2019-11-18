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
algo_list=['neural_network','nearest_neighbors','random_forest','xgboost']

ll=1
start_row=0
n_processes=32

if data=='motion_sense':
	n_test_samples=30
	n_valid_samples=30
	n_train_samples=300
	embedding_list=['time','raw','lead_lag','rectilinear']
	order_dict={
		'raw':(np.arange(5)+1),'time':np.arange(4)+1,'lead_lag':np.arange(3)+1,
		'rectilinear':np.arange(5)+1}

elif data=='urban_sound':
	n_test_samples=500
	n_valid_samples=500
	n_train_samples=4435
	embedding_list=['time','raw','lead_lag']
	order_dict={
		'raw':np.array([2,6,14,30,62,126,254,510]),'time':np.arange(12)+1,
		'lead_lag':np.arange(8)+1}

elif data=='quick_draw':
	n_train_samples=200*340
	n_valid_samples=20*340
	n_test_samples=20*340
	embedding_list=[
		'time','raw','lead_lag','rectilinear','stroke_1','stroke_2','stroke_3']
	order_dict={
		'raw':(np.arange(12)+1),'time':np.arange(8)+1,'lead_lag':np.arange(6)+1,
		'rectilinear':np.arange(12)+1,'stroke_1':np.arange(8)+1,
		'stroke_2':np.arange(8)+1,'stroke_3':np.arange(8)+1}

results_df=pd.DataFrame(
	{'accuracy':[],'embedding':[],'algo':[],'order':[],'n_features':[]})


# Load input data 
for embedding in embedding_list:
	for order in order_dict[embedding]:
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

		for algo in algo_list:
			# Fit and evaluate algorithm
			learnSig=LearnSig(algo,inputSig,n_processes=n_processes)
			learnSig.train(train_X,train_y,valid_X=valid_X,valid_y=valid_y)
			test_results=learnSig.evaluate(test_X,test_y,metrics=['accuracy'])
			print(test_results)

			results_df=results_df.append(
				{'accuracy':test_results['accuracy'],'embedding':embedding,
				'algo':algo,'order':order,'n_features':train_X.shape[1]},
				ignore_index=True)
			print(results_df)


			# Write the results in a separate text file.

			all_results_dir=os.path.join(results_dir,'results_all_files')
			if not os.path.exists(all_results_dir):
				os.mkdir(all_results_dir)

			results_path=os.path.join(
				all_results_dir,learnSig.model_name+".txt")
			file = open(results_path,"w")

			file.write("Dataset: %s \n" % (data))
			file.write("Algorithm: %s \n" % (algo))
			file.write("Embedding: %s \n" %(embedding))
			file.write("Signature order: %i \n" % (order)) 

			file.write("Number of training samples: %i \n" % (n_train_samples))
			file.write(
				"Number of validation samples: %i \n" % (n_valid_samples))
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

results_df.to_csv(
	os.path.join(results_dir,'%s_embedding_comparison.csv'%(data)))




