from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.metrics import top_k_categorical_accuracy
def top_3_accuracy(x,y):return top_k_categorical_accuracy(x,y,3)


def dense_model_1(n_features,n_classes,lr=1):
	model = Sequential()
	model.add(Dropout(0.5,input_shape=(n_features,)))
	model.add(Dense(units=64, activation='linear', input_shape=(n_features,)))
	model.add(Dense(n_classes, activation = 'softmax'))
	opt=SGD(lr=lr)
	model.compile(optimizer = opt, 
	                          loss = 'categorical_crossentropy', 
	                          metrics = ['categorical_accuracy', top_3_accuracy])
	#model.name="dense_model_1"
	model.summary()
	return(model)


def dense_model_2(n_features,n_classes):
	model = Sequential()
	model.add(Dense(units=256, activation='relu', input_shape=(n_features,)))
	model.add(Dense(units=256, activation='relu'))
	model.add(Dense(units=256, activation='relu'))
	model.add(Dense(n_classes, activation = 'softmax'))
	opt=Adam()
	model.compile(optimizer = opt, 
	                          loss = 'categorical_crossentropy', 
	                          metrics = ['categorical_accuracy', top_3_accuracy])
	#model.name="dense_model_2"
	model.summary()
	return(model)