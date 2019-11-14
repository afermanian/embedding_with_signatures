import iisignature as isig
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import os
from ast import literal_eval
from glob import glob
from multiprocessing import Pool
from functools import partial


#data_dir='/volumes/fermanian_disk/Data/embedding' # path to data repository
data_dir='/users/home/fermanian/embedding/Quick-draw-signature'


def create_rectilinear(path):
	""" Return rectilinear embedding from an initial path.

	Parameters
	----------
	path: array, shape (n_points,d)
		The array storing n_points coordinates in R^d that constitute a 
		piecewise linear path.

	Returns
	-------
	new_path: array shape ((n_points-1)*d+1,d)
		The array storing the coordinates in R^d that constitute the
		rectilinear embedding of the initial path.
	"""
	n_points=np.shape(path)[0]
	dim=np.shape(path)[1]
	new_path=np.zeros(((n_points-1)*dim+1,dim))
	new_path[0,:]=path[0,:]
	for i in range(1,n_points):
		new_path[(i-1)*dim+1:i*dim+1,:]=path[i-1,:]
		for j in range(dim):
			new_path[(i-1)*dim+j+1:i*dim+1,j]=path[i,j]
	return(new_path)


def create_lead_lag(path,ll):
	""" Return lead lag embedding from initial path.

	Parameters
	----------
	path: array, shape (n_points,d)
		The array storing the n_points coordinates in R^d that constitute a 
		piecewise linear path.

	ll: int
		The number of lags.

	Returns
	-------
	new_path: array shape (n_points+ll,ll*d +1)
		The array storing the coordinates in R^d that constitute the
		lead lag embedding of the initial path.
	"""

	last_values=path[-1,:]
	path=np.vstack((path,np.repeat([last_values],ll,axis=0)))
	n_points=np.shape(path)[0]
	dim=np.shape(path)[1]
	for k in range(ll):
		for i in range(dim):
			path=np.hstack((path,np.zeros((n_points,1))))
			path[k+1:,-1]=path[:-k-1,i]
	#path=np.hstack((path,np.linspace(0,1,num=n_points).reshape((
	#n_points,1))))/get_curve_length(path)
	path=np.hstack((path,np.linspace(0,1,num=n_points).reshape((n_points,1))))
	return path


class InputSig:
	"""Input object that reads and formats one among 3 datasets for learning
	with signatures.

	Parameters
	----------
	data: {'quick_draw', 'urban_sound', 'motion_sense'}
		Dataset used.

	embedding: {'raw','rectilinear','time','time_rectilinear','lead_lag'}
		Embedding chosen to compute signatures, described in the article ?????.
		If data is 'quick_draw', the values {'stroke_1','stroke_2','stroke_3'}
		are also possible.


	order: int
		Truncation order of the signature.

	ll: int, optional (default None)
		If embedding is 'lead_lag', the number of lags chosen.

	Attributes
	----------
	word_encoder: object of Class LabelEncoder()
		Stores the classes of the dataset chosen.
	"""

	def __init__(self,data,embedding,order,ll=None):
		self.data=data
		self.embedding=embedding
		self.order=order

		self.word_encoder=LabelEncoder()
		self.ll=ll

		if self.data=='quick_draw':
			self.fit_word_encoder(n_samples=10*340)
		else:
			self.fit_word_encoder()

	def get_inputs(self,n_samples=10*340,start_row=0):
		""" Returns a data frame and a vector of labels for each dataset.

		Parameters
		----------
		n_samples: int, default=10
			Number of samples to output.

		start_row: int, default=0
			Row number at which we start reading inputs.

		Returns
		-------
		df: pandas dataframe, shape (n_samples,)
			Dataframe with one column named 'file' that contains the file
			argument of path_to_sig for each sample.

		y: array, shape (n_samples,1)
			Array of labels from word_encoder corresponding to each row of df.
		"""
		
		if self.data=='motion_sense':
			df=pd.read_csv(
				os.path.join(
					data_dir,'MotionSense/motion_sense_shuffled_paths.csv'),
				nrows=n_samples, skiprows=start_row,index_col=0)
			df.columns=['file','Class']

		elif self.data=='urban_sound':
			base_dir =os.path.join(data_dir,'urban-sound-classification/train')
			df = pd.read_csv(os.path.join(base_dir,'train.csv'),nrows=n_samples,

				skiprows=start_row)
			df.columns=['ID','Class']
			df['file'] = df['ID'].apply(lambda x: base_dir+'/Train/'+str(x)+
				'.wav')

		elif self.data=='quick_draw':
			n_samples=n_samples//340
			all_train_paths = glob(
				os.path.join(data_dir,'input','train_simplified','*.csv'))
			out_df_list = []
			for c_path in all_train_paths:
				c_df = pd.read_csv(c_path, nrows=n_samples, skiprows=start_row)
				c_df.columns=['countrycode', 'file', 'key_id', 'recognized', 
				'timestamp', 'Class']
				out_df_list += [c_df[['file', 'Class']]]
			df = pd.concat(out_df_list)

		print(df.head())
		y=df['Class']
		return(df,y)

	def fit_word_encoder(self,n_samples=None):
		""" Fit word encoder with labels of the dataset.

		Parameters
		----------
		n_samples: int, default=None
			Number of samples to load to get the labels.

		Returns
		-------
		self: object
			Returns an instance of self.
		"""

		if os.path.exists('classes_%s.npy' %(self.data)):
			self.word_encoder.classes_ = np.load(
				'classes_'+self.data+'.npy',allow_pickle=True)
		else:
			df,y=self.get_inputs(n_samples=n_samples)
			self.word_encoder.fit(y)
			print(
				"Word encoder fitted with classes: ", 
				self.word_encoder.classes_)
			np.save('classes_%s.npy' %(self.data),self.word_encoder.classes_)

	def data_to_path(self,file):
		"""Read one sample and outputs its path with the embedding selected by
		self.embedding

		Parameters
		----------
		file: str
			- If data='quick_draw', it is the string containing the raw drawing
			coordinates.
			- If data='urban_sound' or 'motion_sense', it is the path to the
			sample file.

		Returns
		-------
		path: array, shape (n_points,d)
			Array with points in R^d that should be linearly interpolated to get
			the embedding.
		"""


		if self.data=='motion_sense':
			raw_data = pd.read_csv(os.path.join(data_dir,file))
			raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
			signal=np.array(raw_data.values)
		
		elif self.data=='urban_sound':
			signal = sf.read(os.path.join(data_dir,file))[0]
			# Normalize the signal to one channel by averaging the two channels
			# when necessary.
			if len(signal.shape)==1:
				signal=signal.reshape((signal.shape[0],1))
			else:
				signal=(signal[:,0]+signal[:,1])/2
				signal=signal.reshape((signal.shape[0],1))
		
		elif self.data=='quick_draw':
			stroke_vec = literal_eval(file) 
			if self.embedding=='stroke_1':
				in_strokes = [(xi,yi,i+1)  for i,(x,y) in enumerate(stroke_vec) 
					for xi,yi in zip(x,y)]
				signal_path=np.stack(in_strokes)
			elif self.embedding=='stroke_2':
				in_strokes=[]
				for i,(x,y) in enumerate(stroke_vec):
					if i!=0:
						in_strokes+=[(x[0],y[0],2*i)]
					for xi,yi in zip(x,y):
						in_strokes+=[(xi,yi,2*i+1)]
					in_strokes+=[(x[-1],y[-1],2*(i+1))]
				signal_path = np.stack(in_strokes[:-1])

			elif self.embedding=='stroke_3':
				in_strokes=[]
				for i,(x,y) in enumerate(stroke_vec):
					t=np.linspace(0,1,num=len(x))
					if i!=0:
						in_strokes+=[(x[0],y[0],3*i-1)]
					for xi,yi,ti in zip(x,y,t):
						in_strokes+=[(xi,yi,3*i+ti)]
					in_strokes+=[(x[-1],y[-1],3*i+2)]
				signal_path = np.stack(in_strokes[:-1])
			else:
				in_strokes = [(xi,yi)  for i,(x,y) in enumerate(stroke_vec) 
					for xi,yi in zip(x,y)]
				signal=np.stack(in_strokes)
		if self.embedding=='lead_lag':
			signal_path=create_lead_lag(signal,self.ll)
		if self.embedding=='raw':
			signal_path=signal
		if self.embedding=='time':
			n_points=np.shape(signal)[0]
			signal_path=np.hstack((signal,np.linspace(0,1,num=n_points).reshape(
				(n_points,1))))
		if self.embedding=='rectilinear':
			signal_path=create_rectilinear(signal)
		if self.embedding=='time_rectilinear':
			n_points=np.shape(signal)[0]
			signal=np.hstack((signal,np.linspace(0,1,num=n_points).reshape((
				n_points,1))))
			signal_path=create_rectilinear(signal)
		return(signal_path)

	def path_to_sig(self,file):
		"""Read one sample and output its signature with the embedding selected
		by self.embedding.

		Parameters
		----------
		file: str
			- If data='quick_draw', it is the string containing the raw drawing
			coordinates.
			- If data='urban_sound' or 'motion_sense', it is the path to the
			sample file.

		Returns
		-------
		sig: array, shape (p)
			Array containing signature coefficients computed on the embedded
			path of the sample corresponding to file.
		"""

		path=self.data_to_path(file)
		sig=isig.sig(path,self.order)
		return(sig)

	def path_to_dyadic_sig(self,file,dyadic_level):
		"""Read one sample and output a vector containing a concatenation of
		signature coefficients. The path is divided into 2^dyadic_levelsubpaths 
		and a signature vector is computed on each subpath. All vectors
		obtained in this way are then concatenated.

		Parameters
		----------
		file: str
			- If data='quick_draw', it is the string containing the raw drawing
			coordinates.
			- If data='urban_sound' or 'motion_sense', it is the path to the
			sample file.

		dyadic_level: int
			It is the level of dyadic partitions considered. The path is divided
			into 2^dyadic_level subpaths and signatures are computed on each
			subpath.

		Returns
		-------
		sig: array, shape (p)
			A signature vector containing all signature coefficients. It is of
			shape p=2^dyadic_level*self.get_sig_dimension().
		"""
		path=self.data_to_path(file)
		n_points=np.shape(path)[0]
		n_subpaths=2**dyadic_level
		window_size=n_points//n_subpaths
		
		if n_subpaths>n_points:
			path=np.vstack(
				(path,np.zeros((n_subpaths-n_points,np.shape(path)[1]))))
			window_size=1
		siglength=self.get_sig_dimension()
		sig=np.zeros(n_subpaths*siglength)
		for i in range(n_subpaths):
			if i==n_subpaths-1:
				subpath=path[i*window_size:,:] 
			else:
				subpath=path[i*window_size:(i+1)*window_size,:]
			sig[i*siglength:(i+1)*siglength]=isig.sig(subpath,self.order)
		return(sig)


	def get_embedding_dimension(self):
		"""Returns the dimension of the output space of the embedded data.

		Returns
		-------
		dimension: int
			The dimension of the output space
		"""
		
		if self.data=='motion_sense':
			d=12
		elif self.data=='urban_sound':
			d=1
		elif self.data=='quick_draw':
			d=2

		if self.embedding=='lead_lag':
			return(self.ll*d +1)
		elif self.embedding=='raw' or self.embedding=='rectilinear':
			return(d)
		else:
			return(d+1)

	def get_sig_dimension(self):
		""" Returns the dimension of the signature vector truncated at
		self.order and with the embedding chosen by self.path

		Returns
		-------
		siglength: int
			The dimension of the signature vector.
		"""
		return(isig.siglength(self.get_embedding_dimension(),self.order))



def unwrap_path_to_sig(file,inputSig,dyadic_level=None):
	""" Unwrap the functions path_to_sig and path_to_dyadic_sig from the object
	inputSig to allow parallelization.

	Parameters
	----------
	file: str
		First argument of inputSig.path_to_sig.
			- If data='quick_draw', it is the string containing the raw drawing
			coordinates.
			- If data='urban_sound' or 'motion_sense', it is the path to the
			sample file.

	inputSig: instance of the class InputSig
		The obejct containing information about which data we are working
		on and with which embedding.

	dyadic_level: int, default=None
		If not None, the function path_to_dyadic_sig is used and dyadic_level is
		then the level of the dyadic partition considered. Otherwise the
		function path_to_sig is outputed.

	Returns
	-------
	sig: array, shape (p)
		The signature vector, output of path_to_sig or path_to_dyadic_sig.

	"""
	if dyadic_level:
		return(inputSig.path_to_dyadic_sig(file,dyadic_level))
	else:
		return(inputSig.path_to_sig(file))
	


def get_input_X_y(inputSig,n_samples,start_row,n_processes=1,dyadic_level=None):
	""" Return training input matrix X and output y corresponding to inputSig
	for n_samples samples.

	Parameters
	----------
	inputSig: instance of the class InputSig
		The object containing information about which data we are working
		on and with which embedding.
	
	n_samples: int
		The number of samples in X and y.

	start_row: int
		The row to start loading data.

	n_processes: int
		Number of parallel processes the signature computations are done with.

	dyadic_level: int, default=None
		If not None, the path is divided into 2^dyadic_level subpaths and
		signatures are computed on each subpath. Otherwise signatures are
		computed on the whole path.

	Returns
	-------
	X: array, shape (n_samples, p)
		Array with samples in rows and signature coefficients in columns.

	y: array, shape (n_samples,1)
		Array of sample labels.
	"""
	df,y=inputSig.get_inputs(n_samples=n_samples,start_row=start_row)
	pool=Pool(processes=n_processes)
	data_map=partial(
		unwrap_path_to_sig,inputSig=inputSig,dyadic_level=dyadic_level)
	df['signature']=pool.map(data_map,df['file'])
	X= np.stack(df['signature'], 0)
	y=inputSig.word_encoder.transform(df['Class'])
	return(X,y)
	




