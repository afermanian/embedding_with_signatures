import numpy as np
import os
import time
import sys

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,TensorBoard
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.metrics import top_k_categorical_accuracy
def top_3_accuracy(x,y): return top_k_categorical_accuracy(x,y,3)

from sklearn.metrics import accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from neural_network_models import dense_model_1


class LearnSig:
	""" Object that fits and predicts a learning algorithm with signatures.

	Parameters
	----------
	algo: {'neural_network','random_forest','nearest_neighbors','xgboost'}
		Name of the algorithm used. See ???? for details.

	inputSig: instance of the class InputSig
		Object containing information about data, embedding, truncation order
		and the methods to load data.

	n_processes: int, default=1
		Number of parallel processes used to fit the algorithm when
		parallelization is possible.

	Attributes
	----------
	model_name: str
		The name results and fitted models are saved under.
	params: dict
		Hyperparameters given to the algorithm. The default values are the one
		used in ????.
	"""
	def __init__(self,algo,inputSig,n_processes=1):
		self.algo=algo
		self.n_processes=n_processes
		self.inputSig=inputSig
		if self.inputSig.embedding=='lead_lag':
			self.model_name="%s_%s_%s_embedding_lag_%i_order_%i_%s" % (
				self.inputSig.data,self.algo,self.inputSig.embedding,
				self.inputSig.ll,self.inputSig.order,
				time.strftime("%m%d-%H%M%S"))
		else:
			self.model_name="%s_%s_%s_embedding_order_%i_%s" % (
				self.inputSig.data,self.algo,self.inputSig.embedding,
				self.inputSig.order,time.strftime("%m%d-%H%M%S"))

		if self.algo=='neural_network':
			self.params={'lr':1,'batch_size':128,'epochs':200}
		elif self.algo=='random_forest':
			self.params={'n_trees':50}
		elif self.algo=='xgboost':
			self.params={
				'objective':'multi:softmax','silent': 1,
				'nthread':self.n_processes,'eval_metric':'merror',
				'verbosity':3, 
				'num_class':len(self.inputSig.word_encoder.classes_),
				'max_depth':3,'gamma':0.5,'num_round':100}
		else:
			self.params={}
	
	def train(
			self,train_X,train_y,valid_X=None,valid_y=None,params=None,
			results_dir='results'):
		""" Train the algorithm.

		Parameters
		----------
		train_X: array, shape (n_samples, siglength)
			Training data.

		train_y: array, shape (n_samples,1)
			Target values: labels.

		valid_X: array, shape (n_valid_samples, siglength), default=None
			Validation data, necessary if algo='neural_network' or 'xgboost'.

		valid_y: array, shape (n_valid_samples,1), default=None
			Validation target values, necessary if algo='neural_network' or
			'xgboost'.

		params: dict, default=None
			Parameters of the algorithm. If None, self.params is used, otherwise
			self.params is overwritten with values in params.

		results_dir: str, default='results'
			Path to the directory where the fitted model is saved.

		Results
		-------

		model: object
			An instance of the fitted model. It is also stored in self.model.
		"""
		
		if params:
			self.params=params

		

		print("Training data shape: ",train_X.shape)
		if self.algo=='neural_network':
			train_y_cat=to_categorical(train_y)
			valid_y_cat=to_categorical(valid_y)

			nb_features=train_X.shape[1]
			nb_classes=train_y_cat.shape[1]

			# Define callbacks
			checkpoint = ModelCheckpoint(
				os.path.join(results_dir,self.model_name+".h5"), 
				monitor='val_loss',verbose=1,save_best_only=True, mode='min')
			tensorBoard=TensorBoard(log_dir=os.path.join('./logs',
				self.model_name))
			lr_update= ReduceLROnPlateau(monitor='val_loss', factor=0.5,
			 patience=10,verbose=1)
			callbacks_list = [checkpoint,tensorBoard,lr_update]

			# Load model
			self.model=dense_model_1(nb_features,nb_classes,lr=self.params
				['lr'])

			self.model.fit(
				train_X, train_y_cat,validation_data = (valid_X, valid_y_cat), 
				batch_size = self.params['batch_size'],
				epochs = self.params['epochs'],callbacks=callbacks_list)

		elif self.algo=='nearest_neighbors':
			self.model=KNeighborsClassifier(n_jobs=self.n_processes)
			self.model.fit(train_X,train_y)

		elif self.algo=='random_forest':
			self.model=RandomForestClassifier(
				n_jobs=self.n_processes,n_estimators=self.params['n_trees'])
			self.model.fit(train_X,train_y)

		else:
			dtrain=xgb.DMatrix(train_X,label=train_y)
			dval=xgb.DMatrix(valid_X,label=valid_y)

			evallist = [(dtrain, 'train'),(dval, 'eval')]
			self.model=xgb.train(
				self.params,dtrain, self.params['num_round'],evallist,
				early_stopping_rounds=20)
			self.model.save_model(os.path.join(results_dir,self.model_name))
		
		return(self.model)

	def evaluate(self,test_X,test_y,metrics=['accuracy']):
		"""Evaluate the fitted model.

		Parameters
		----------
		test_X: array, shape (n_samples,siglength)
			Test data to compute metrics on.

		test_y: array, shape (n_samples,1)
			Test true labels.

		metrics: list of str, default=['accuracy']
			List of metric names, the accepted metrics are 'accuracy' or
			'f1_score'

		Returns
		-------
		test_results: dict
			Dictionary of results, the keys are the elements in metrics. For
			examples, test_results['accuracy'] contains the test accuracy score.
		"""
		if self.algo=='neural_network':
			pred_y=self.model.predict_classes(test_X)
		elif self.algo=='xgboost':
			dtest=xgb.DMatrix(test_X)
			pred_y = self.model.predict(
				dtest,ntree_limit=self.model.best_ntree_limit)
		else:
			pred_y = self.model.predict(test_X)

		test_results={}
		if 'accuracy' in metrics:
			test_results['accuracy']=accuracy_score(test_y,pred_y)
		if 'f1_score' in metrics:
			test_results['f1_score']=f1_score(test_y,pred_y,average='macro')
		
		return(test_results)












