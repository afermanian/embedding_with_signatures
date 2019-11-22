import os
import sys

import time
from keras.callbacks import ModelCheckpoint,TensorBoard
from glob import glob
import pandas as pd
import numpy as np

from quick_draw_generator import QuickDrawGenerator
from neural_network_models import dense_model_2
from read_format import InputSig,get_input_X_y



from keras.models import load_model
from keras.metrics import top_k_categorical_accuracy
def top_3_accuracy(x,y): return top_k_categorical_accuracy(x,y,3)


#data_dir='/volumes/fermanian_disk/Data/embedding' # path to data repository
data_dir='/users/home/fermanian/embedding/Quick-draw-signature'


# Hyperparameters
order=6
ll=1
n_processes=32
batch_size=128
epochs=300
initial_epoch=0

n_train_samples=1792*340
n_valid_samples=256*340
n_test_samples=256*340
n_max_train_samples=35840*340

first_row=0

n_tot=(n_max_train_samples+n_test_samples+n_valid_samples)//340
print('Total number of samples per class : ',n_tot)


# model_name= "quick_draw_dense_model_2_lead_lag_embedding_lag_%i_order_%i_%s" % (
# 	ll,order,time.strftime("%m%d-%H%M%S"))

# model_path=os.path.join('results',model_name+"_{epoch:02d}-{val_loss:.2f}.h5")
# log_path=os.path.join('./logs',model_name)
# model_param_path=os.path.join('results',model_name+".txt")
# print(model_param_path)

# # Define callbacks
# checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1,
# 	save_best_only=False, mode='min')
# tensorBoard=TensorBoard(log_dir=log_path)
# callbacks_list = [checkpoint,tensorBoard]

# # Get input data
# all_train_paths=glob(os.path.join(data_dir,'input','train_simplified', '*.csv'))
# cache={
# 	i:pd.read_csv(all_train_paths[i],
# 	names=['countrycode', 'file', 'key_id', 'recognized', 'timestamp', 'Class'],
# 	nrows=n_tot,header=0) for i in range(len(all_train_paths))}
# train_generator=QuickDrawGenerator(
# 	n_train_samples,n_max_train_samples,batch_size,first_row,cache,order=order)
# valid_generator=QuickDrawGenerator(
# 	n_valid_samples,n_valid_samples,batch_size,first_row+n_max_train_samples,
# 	cache,order=order)
# test_generator=QuickDrawGenerator(
# 	n_test_samples,n_test_samples,batch_size,
# 	first_row+n_max_train_samples+n_valid_samples,cache,order=order)

# # Get model
# n_features=train_generator.inputSig.get_sig_dimension()
# n_classes=len(train_generator.word_encoder.classes_)

# model=dense_model_2(n_features,n_classes)
# print(model.name)

# print("fitting")
# # Fit model

# model.fit_generator(generator=train_generator,
#                    epochs = epochs,
#                    verbose = 1,
#                    validation_data = valid_generator,
#                    callbacks=callbacks_list,
#                    use_multiprocessing=True,
#                    workers=n_processes,
#                    max_queue_size=128)

# print("predicting")
# model_results = model.evaluate_generator(test_generator)
# print(model_results)


def mapk(y_true,y_pred,k=3):
	"""
	Computes the mean average precision at k

	Parameters:
	y_true: numpy array with true labels, of size n number of samples
	y_pred: numpy array of k ranked predictions

	"""

	n=np.shape(y_true)[0]
	score=0
	for i in range(n):
		continue_var=True
		for j in range(k):
			if (y_true[i] in y_pred[i,:(j+1)]) and continue_var:
				score+=1/(j+1)
				continue_var=False
	return(score/n)

initial_model_path=str(sys.argv[1])
trained_path=os.path.join('results',initial_model_path)
model_param_path=os.path.join('results',initial_model_path+".txt")

print("Get input sig ")
inputSig=InputSig('quick_draw','lead_lag',6,ll=1)

test_X,test_y=get_input_X_y(
			inputSig,n_test_samples,first_row+n_max_train_samples+n_valid_samples,
			n_processes=n_processes)
norm_vec=np.load('norm_vec_quick_draw_generator.npy')
test_X=test_X/norm_vec

model=load_model(trained_path,custom_objects={'top_3_accuracy':top_3_accuracy})

print("Predict")
pred_y_cat=model.predict(test_X,batch_size=batch_size)
top_3_pred =np.vstack([inputSig.word_encoder.classes_[np.argsort(-1*c_pred)[:3]] for c_pred in pred_y_cat])
print(top_3_pred.shape)

test_labels=inputSig.word_encoder.inverse_transform(test_y)
print(top_3_pred[0,:],test_labels[0])
print(test_labels[0] in top_3_pred[0,:])

mapk=mapk(test_labels,top_3_pred)
print("MAPK at 3: ",mapk)

# Save results
file = open(model_param_path,"w")
file.write("Signature order: %i \n" % (order)) 
file.write("Epochs: %i \n" % (epochs)) 
file.write("Batch size: %i \n" %(batch_size))
file.write("Optimizer: Adam \n")
file.write("Number of training samples: %i \n" % (n_train_samples))
file.write("Number of validation samples: %i \n" % (n_valid_samples))
file.write("Number of test samples: %i \n" % (n_test_samples))
file.write("Number of max training samples: %i \n" % (n_max_train_samples))
file.write("First row of training data: %i \n" % (first_row))
file.write("Lag %s \n" % (ll,))
file.write("Model chosen: \n")
file.write("Model name:%s \n" %(model.name,))
model.summary(print_fn=lambda x: file.write(x + '\n'))
file.write("Mapk at 3: %s" % (mapk))
file.write("Top 3 accuracy: %s" % (top_3))
file.close() 





