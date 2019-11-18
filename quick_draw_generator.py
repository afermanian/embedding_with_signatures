from keras.utils import Sequence
import numpy as np
from sklearn.preprocessing import LabelEncoder

from read_format import InputSig



class QuickDrawGenerator(Sequence):
	def __init__(self, n_files,n_max_samples,batch_size,first_row,cache,order=6):
		self.word_encoder=LabelEncoder()
		self.word_encoder.classes_ = np.load('classes_quick_draw.npy',allow_pickle=True)

		self.batch_size = batch_size
		self.first_row=first_row
		self.n_files=n_files//340
		self.order=order

		self.cache=cache
		self.n_max_samples=n_max_samples//340

		self.inputSig=InputSig('quick_draw','lead_lag',order,ll=1)
		self.norm_vec=np.load('norm_vec_quick_draw_generator.npy')

		self.on_epoch_end()

	def __len__(self) :
		return (np.ceil(self.n_files*340/float(self.batch_size))).astype(np.int)


	def __getitem__(self, idx) :
		#start_time=time.time()
		batch_start=idx*self.batch_size
		batch_stop=(idx+1)*self.batch_size

		batch_files=self.indexes[batch_start:batch_stop,0]
		batch_indexes=self.indexes[batch_start:batch_stop,1]

		SigX=np.zeros((self.batch_size,self.inputSig.get_sig_dimension()))
		y=np.zeros(self.batch_size,dtype=object)
		for i in range(self.batch_size):
			path_i=self.cache[batch_files[i]].ix[
				self.first_row+batch_indexes[i],'file']
			SigX[i,:]=self.inputSig.path_to_sig(path_i,self.order,1)
			y[i]=self.cache[batch_files[i]].ix[
				self.first_row+batch_indexes[i],'Class']

		SigX=SigX/self.norm_vec
		return SigX,to_categorical(
			self.word_encoder.transform(y),
			num_classes=len(self.word_encoder.classes_))


	def on_epoch_end(self):
		file_indexes=np.repeat(np.arange(340),self.n_files)
		print(file_indexes)
		row_indexes=np.tile(
			np.random.choice(np.arange(self.n_max_samples),size=self.n_files,replace=False),
			len(self.word_encoder.classes_))
		print(row_indexes)

		self.indexes=np.transpose(np.array([file_indexes,row_indexes]))



