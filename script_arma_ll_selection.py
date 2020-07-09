import numpy as np
import time
import sys
import os
import pandas as pd

from read_format import InputSig,get_input_X_y
from train_sig import LearnSig


results_dir='results'

start_time=time.time()

ar_params=[1,.9]
ar=len(ar_params)-1
data='arma'
arma_params={'ar':ar_params,'ma':None,'length':100,'noise_std':1}
algo='linear_regression'

start_row=0
n_processes=2
n_iter=20

#n_train_values=np.floor(np.linspace(start=5,stop=1000,num=5)).astype(int)
n_train_values=[1000]
print(n_train_values)
n_test_values=n_train_values

embedding='lead_lag'
#order_dict={
#		'raw':(np.arange(12)+1),'time':np.arange(8)+1,'lead_lag':np.arange(6)+1,
#		'rectilinear':np.arange(12)+1,'stroke_1':np.arange(8)+1,
#		'stroke_2':np.arange(8)+1,'stroke_3':np.arange(8)+1}

ll_list=np.arange(7)
order_dict={
		'0':np.arange(10)+1,'1':np.arange(7)+1,'2':np.arange(5)+1,
		'3':np.arange(5)+1,'4':np.arange(4)+1,'5':np.arange(4)+1,
		'6':np.arange(3)+1}


results_df=pd.DataFrame()


def simu_fit_test(arma_params,ll,order,n_train_samples,n_test_samples,data='arma',embedding='lead_lag'):
	inputSig=InputSig(data,embedding,order,ll=ll,arma_params=arma_params)

	train_X,train_y=get_input_X_y(
		inputSig,n_train_samples,start_row,n_processes=n_processes)
	max_train_X=np.amax(np.absolute(train_X),axis=0)
	train_X=train_X/max_train_X

	test_X,test_y=get_input_X_y(
		inputSig,n_test_samples,start_row+n_train_samples,
		n_processes=n_processes)
	test_X=test_X/max_train_X

	# Fit and evaluate algorithm
	learnSig=LearnSig(algo,inputSig,n_processes=n_processes)
	learnSig.train(train_X,train_y)
	test_results=learnSig.evaluate(test_X,test_y,metrics=['error_l2'])

	return(test_results['error_l2'])



# Load input data
for i in range(n_iter):
	print("Iteratio number : ",i)
	for j in range(len(n_train_values)):
		n_train_samples=n_train_values[j]
		n_test_samples=n_test_values[j]
		print("Sample size : ",n_train_samples)
		for ll in ll_list:
			print("Lag : ",ll)
			error_l2_ll=[]
			for order in order_dict[str(ll)]:
				print("Truncation order : ",order)
				try:
					error_l2_ll.append(simu_fit_test(arma_params,ll,order,n_train_samples,n_test_samples))
				except:
					error_l2_ll.append(10**15)

			print(error_l2_ll)
			best_order=order_dict[str(ll)][np.argmin(error_l2_ll)]
			best_order_error=simu_fit_test(arma_params,ll,best_order,n_train_samples,n_test_samples)

			results_df=results_df.append(
					{'error_l2':best_order_error,'embedding':embedding,
					'algo':algo,'order':best_order,'n_iter':i,
					'll':ll,'sample size':n_train_samples,'ar':ar},ignore_index=True)

results_df.to_csv(
	os.path.join(results_dir,'%s_ll_selection_ar_%i.csv'%(data,ar)))
