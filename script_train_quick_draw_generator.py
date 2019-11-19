import os
import time
from keras.callbacks import ModelCheckpoint,TensorBoard
from glob import glob
import pandas as pd

from quick_draw_generator import QuickDrawGenerator
from neural_network_models import dense_model_2

#data_dir='/volumes/fermanian_disk/Data/embedding' # path to data repository
data_dir='/users/home/fermanian/embedding/Quick-draw-signature'



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


model_name= "quick_draw_dense_model_2_lead_lag_embedding_lag_%i_order_%i_%s" % (
	ll,order,time.strftime("%m%d-%H%M%S"))

model_path=os.path.join('results',model_name+"_{epoch:02d}-{val_loss:.2f}.h5")
log_path=os.path.join('./logs',model_name)
model_param_path=os.path.join('results',model_name+".txt")
print(model_param_path)

# Define callbacks
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1,
	save_best_only=False, mode='min')
tensorBoard=TensorBoard(log_dir=log_path)
callbacks_list = [checkpoint,tensorBoard]

# Get input data

all_train_paths=glob(os.path.join(data_dir,'input','train_simplified', '*.csv'))
cache={
	i:pd.read_csv(all_train_paths[i],
	names=['countrycode', 'file', 'key_id', 'recognized', 'timestamp', 'Class'],
	nrows=n_tot,header=0) for i in range(len(all_train_paths))}
train_generator=QuickDrawGenerator(
	n_train_samples,n_max_train_samples,batch_size,first_row,cache,order=order)
valid_generator=QuickDrawGenerator(
	n_valid_samples,n_valid_samples,batch_size,first_row+n_max_train_samples,
	cache,order=order)
test_generator=QuickDrawGenerator(
	n_test_samples,n_test_samples,batch_size,
	first_row+n_max_train_samples+n_valid_samples,cache,order=order)

# Get model

n_features=train_generator.inputSig.get_sig_dimension()
n_classes=len(train_generator.word_encoder.classes_)

model=dense_model_2(n_features,n_classes)
print(model.name)

print("fitting")
# Fit model

model.fit_generator(generator=train_generator,
                   epochs = epochs,
                   verbose = 1,
                   validation_data = valid_generator,
                   callbacks=callbacks_list,
                   use_multiprocessing=True,
                   workers=n_processes,
                   max_queue_size=128)


print("--- %s seconds ---" % (time.time() - start_time))
print("predicting")
model_results = model.evaluate_generator(test_generator)
print(model_results)

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
file.write("Number of features: %s \n" % (n_features,))
file.write("Lag %s \n" % (ll,))
file.write("Model chosen: \n")
file.write("Model name:%s \n" %(model.name,))
model.summary(print_fn=lambda x: file.write(x + '\n'))
file.write("Time %s \n" % (time.time() - start_time))
file.write("Model_results: %s" %(model_results))
file.close() 





