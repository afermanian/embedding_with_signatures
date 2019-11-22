import numpy as np
import sys 
import os

from keras.models import load_model
from keras.metrics import top_k_categorical_accuracy
def top_3_accuracy(x,y): return top_k_categorical_accuracy(x,y,3)

from read_format import get_quick_draw_submission_input_X,InputSig


initial_model_path=str(sys.argv[1])
trained_path=os.path.join('results',initial_model_path)
submission_path=os.path.join('results',initial_model_path+'_submission.csv')

batch_size=128
order=6
ll=1
n_processes=2

model=load_model(trained_path,custom_objects={'top_3_accuracy':top_3_accuracy})


# Get data
inputSig=InputSig('quick_draw','lead_lag',order,ll=ll)
test_X=get_quick_draw_submission_input_X(inputSig,n_processes=n_processes)
norm_vec=np.load('norm_vec_quick_draw_generator.npy')
test_X=test_X/norm_vec

print(test_X.shape)


pred_y_cat=model.predict(test_X,batch_size=batch_size)
print(pred_y_cat.shape)

top_3_pred =[inputSig.word_encoder.classes_[np.argsort(-1*c_pred)[:3]] for c_pred in pred_y_cat]
#top_3_pred =[test_generator.word_encoder.classes_[np.argsort(-1*c_pred)[:3]] for c_pred in pred_y_cat]

top_3_pred = [' '.join([col.replace(' ', '_') for col in row]) for row in top_3_pred]
test_df['word'] = top_3_pred
test_df[['key_id', 'word']].to_csv(submission_path, index=False)


