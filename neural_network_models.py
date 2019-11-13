from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.metrics import top_k_categorical_accuracy
def top_3_accuracy(x,y):return top_k_categorical_accuracy(x,y,3)


def dense_model_1(nb_features,nb_classes,lr=0.001):
	model = Sequential()
	model.add(Dropout(0.5,input_shape=(nb_features,)))
	model.add(Dense(units=64, activation='linear', input_shape=(nb_features,)))
	model.add(Dense(nb_classes, activation = 'softmax'))
	opt=SGD(lr=lr)
	model.compile(optimizer = opt, 
	                          loss = 'categorical_crossentropy', 
	                          metrics = ['categorical_accuracy', top_3_accuracy])
	model.name="dense_model_1"
	model.summary()
	return(model)