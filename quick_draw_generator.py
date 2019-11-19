from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.preprocessing import LabelEncoder

from read_format import InputSig



class QuickDrawGenerator(Sequence):
	''' The object generating batches to train a neural network. For each
	batch, it loads the data, embeds it and computes signatures. At each
	epoch, n_samples samples are taken randomly among n_max_samples.

	Parameters
	----------

	n_samples: int
		Number of samples per epoch. n_samples//340 are selected for each of the
		340 classes.

	n_max_samples: int
		Number of samples among which n_samples are randomly selected.
		n_max_samples must be larger than n_samples.

	batch_size: int
		The batch size on which each gradient descent step is done.

	first_row: int
		For each class, first_row//340 is the first row from which
		samples are taken. 

	cache: dict
		A dictionary with all csv files loaded as pandas data frame to speed up
		file reading. The keys are integers from 0 to 339 and cache[i]
		is one pandas data frame corresponding to one class

	order: int, default=6
		The order of truncation of the signature.

	Attributes
	----------

	word_encoder: object of Class LabelEncoder()
		Stores the classes of the Quick, Draw! dataset.

	inputSig: object of Class InputSig()
		An instance of InputSig() that enables to compute signatures and
		contains information about the emnbedding and signatures.

	norm_vec: array, shape (p,)
		Normalization vector saved by the script
		preprocessing_generator_quick_draw that contains the maximum of each
		signature coefficient over the data. Signatures are divided by norm_vec
		so that every coefficient lies in [-1,1].

	'''
	def __init__(self, n_samples,n_max_samples,batch_size,first_row,cache,order=6):
		self.word_encoder=LabelEncoder()
		self.word_encoder.classes_ = np.load('classes_quick_draw.npy',allow_pickle=True)

		print("Number of classes",len(self.word_encoder.classes_))
		self.batch_size = batch_size
		self.first_row=first_row//340
		self.n_samples=n_samples//340
		self.n_max_samples=n_max_samples//340

		self.cache=cache

		self.inputSig=InputSig('quick_draw','lead_lag',order,ll=1)
		self.norm_vec=np.load('norm_vec_quick_draw_generator.npy')

		self.on_epoch_end()

	def __len__(self) :
		""" Returns the number of steps made in each epoch, that is the number
		of batches.
		"""
		return (np.ceil(self.n_samples*340/float(self.batch_size))).astype(np.int)


	def __getitem__(self, idx) :
		""" Returns inputs for one batch training.

		Parameters
		----------
		idx: int,
			Index of the batch.

		Returns
		-------
		SigX: array, shape (batch_size,p)
			Matrix of signature coefficients of the batch samples.

		y: array, shape (batch_size,340)
			One-hot encoding of the classes array.
		"""
		#start_time=time.time()
		batch_start=idx*self.batch_size
		batch_stop=(idx+1)*self.batch_size

		batch_files=self.indexes[batch_start:batch_stop,0]
		batch_indexes=self.indexes[batch_start:batch_stop,1]

		SigX=np.zeros((self.batch_size,self.inputSig.get_sig_dimension()))
		y=np.zeros(self.batch_size,dtype=object)
		for i in range(self.batch_size):
			path_i=self.cache[batch_files[i]]['file'][
				self.first_row+batch_indexes[i]]
			SigX[i,:]=self.inputSig.path_to_sig(path_i)
			y[i]=self.cache[batch_files[i]]['Class'][
				self.first_row+batch_indexes[i]]

		SigX=SigX/self.norm_vec
		return (SigX,to_categorical(
			self.word_encoder.transform(y),
			num_classes=len(self.word_encoder.classes_)))


	def on_epoch_end(self):
		""" Generates a matrix with random indexes of samples used in each
		epoch.
		"""
		file_indexes=np.repeat(np.arange(340),self.n_samples)
		row_indexes=np.tile(
			np.random.choice(np.arange(self.n_max_samples),size=self.n_samples,replace=False),
			len(self.word_encoder.classes_))

		self.indexes=np.transpose(np.array([file_indexes,row_indexes]))
		print('Indexes array shape: ',self.indexes.shape)




