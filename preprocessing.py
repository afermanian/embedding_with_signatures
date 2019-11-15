import pandas as pd
import os

#data_dir='/volumes/fermanian_disk/Data/embedding' # path to data repository
data_dir='/users/home/fermanian/embedding/Quick-draw-signature'


def save_motion_sense_shuffled_paths_df():
	'''Create a csv with the path to all Motion Sense files and shuffle them.
	
	Returns
	-------

	df: object of class pandas DataFrame, shape (360,2)
		Dataframe with two columns named 'file' and 'Class' and 360 rows
		corresponding to all Motion Sense samples. 'file' contains the path to
		the file and 'Class' its label.
	'''
	ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
	TRIAL_CODES = {
		ACT_LABELS[0]:[1,2,11],
		ACT_LABELS[1]:[3,4,12],
		ACT_LABELS[2]:[7,8,15],
		ACT_LABELS[3]:[9,16],
		ACT_LABELS[4]:[6,14],
		ACT_LABELS[5]:[5,13]
	}

	ds_list = pd.read_csv(
		os.path.join(data_dir,"MotionSense/data_subjects_info.csv"))
	trial_codes = [TRIAL_CODES[act] for act in ACT_LABELS]
	names_list=[]
	act_list=[]
	for sub_id in ds_list["code"]:
		for act_id, act in enumerate(ACT_LABELS):
			for trial in trial_codes[act_id]:
				names_list.append(
					os.path.join(data_dir,'MotionSense/A_DeviceMotion_data/'+act
						+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'))
				act_list.append(act_id)
	df=pd.DataFrame(data={'file':names_list,'Class':act_list})
	print(df.head())
	print(df.shape)
	df = df.sample(frac=1).reset_index(drop=True)
	print(df.head())
	print(df.shape)
	df[['file','Class']].to_csv(
		os.path.join(data_dir,'MotionSense/motion_sense_shuffled_paths.csv'))
	return(df)

def save_urban_sound_shuffled_paths():
	'''Shuffle the file train.csv containing ID and Class of all Urban Sound
	samples.
	
	Returns
	-------

	df: object of class pandas DataFrame, shape (5435,2)
		Dataframe with two columns named 'ID' and 'Class' and 5435 rows
		corresponding to all Urban Sound samples. 'ID' contains  the ID of the
		sample and 'Class' its label.
	'''
	urban_sound_dir =os.path.join(
		data_dir,'urban-sound-classification/train')
	df = pd.read_csv(os.path.join(urban_sound_dir,'train.csv'))
	print(df.head())
	print(df.shape)
	df = df.sample(frac=1).reset_index(drop=True)
	print(df.head())
	print(df.shape)
	df.to_csv(os.path.join(urban_sound_dir,'shuffled_train.csv'))
	return(df)




save_motion_sense_shuffled_paths_df()
save_urban_sound_shuffled_paths()







