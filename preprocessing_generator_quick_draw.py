import numpy as np

from read_format import InputSig,get_input_X_y


order=6
ll=1
n_processes=32

n_train_samples=1792*340
n_valid_samples=256*340
n_test_samples=256*340
n_max_train_samples=35840*340

n_tot=n_max_train_samples+n_test_samples+n_valid_samples
print('Total number of samples: ',n_tot)


def save_normalization_vector():
	inputSig=InputSig('quick_draw','lead_lag',order,ll=ll)

	batch_size=64*340
	n_iterations=n_tot//batch_size
	print(n_iterations)
	start_row=0
	print(inputSig.get_sig_dimension())
	max_SigX=np.zeros(inputSig.get_sig_dimension())
	print(max_SigX.shape)
	for i in range(n_iterations):
		print(start_row)
		SigX,train_y=get_input_X_y(
			inputSig,batch_size,start_row,n_processes=n_processes)
		max_SigX=np.maximum(np.max(np.absolute(SigX),axis=0),max_SigX)
		start_row+=batch_size
		print(max_SigX[15])
		print(np.max(np.absolute(SigX),axis=0)[15])
	
	np.save('norm_vec_quick_draw_generator.npy',max_SigX)
	return(max_SigX)


save_normalization_vector()
