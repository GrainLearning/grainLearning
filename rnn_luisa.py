import os, h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

import wandb
import IPython

def load_data():
	"""
	Loads the data from hdf5 file only for the 'drained' case.
	Returns a tuple of 2 arrays:
	 outputs: tensor of shape (n_samples,length_sequence,variables)=(4474, 200, 11). 
	          11 variables: 3 inputs, 7 outputs and the confinment (float) in that order.
	 contact_params: contact parameters of each sample (for the 3 different confinements)
	"""
	f = h5py.File('sequences.hdf5', 'r') #binary file containing the consolidating data by Aron
	conf = np.array(list(f.keys()),dtype='float64') #Vector of confinement pressures
	
	contact_params,outputs = ([] for i in range(0,2))
	for k in f.keys(): #Adding info a different confinment pressures to the same list
		#for i in range(0,len(f[k]['drained']['contact_params'])): #adding the confinment to the contact params
		#	contact_params.append(np.append(f[k]['drained']['contact_params'][i],float(k)))
		outputs_i=[]
		for i in range(0,len(f[k]['drained']['outputs'])): #adding the confinment to the outputs
			output_i = f[k]['drained']['outputs'][i]
			conf_array=np.repeat(np.array(float(k)),np.shape(output_i)[0])
			outputs_i.append(np.column_stack((output_i,conf_array)))
		outputs.extend(np.concatenate((f[k]['drained']['inputs'],outputs_i),axis=2))
		contact_params.extend(f[k]['drained']['contact_params'])
	return np.array(outputs),np.array(contact_params)

def load_data_conf(conf):
	"""
	Loads the data from hdf5 file for a single confinment pressure conf and puts it together in lists
	Depreceated. I ca now use load_data() that takes ALL confinement pressures
	"""
	f = h5py.File('rnn_data.hdf5', 'r') #binary file containing the consolidating data by Aron
	inputs=np.array(f[conf]['drained']['inputs'])
	outputs=np.array(f[conf]['drained']['outputs'])
	contact_params=np.array(f[conf]['drained']['contact_params'])
	return np.concatenate([inputs, outputs], axis=2),contact_params

def split_data(inputs_outputs,contact_params,):
	"""
	Splits the data between train, test and validation sets and returns 
	a dict with tuples of the sets
	"""
	test_frac=0.2
	puts_train,puts_test,contact_train,contact_test = train_test_split(inputs_outputs,contact_params,test_size=test_frac,random_state=10,shuffle=True)	
	puts_test,puts_val,contact_test,contact_val = train_test_split(puts_test,contact_test,test_size=0.5,random_state=20,shuffle=True)

	return{
		'train':(puts_train,contact_train),
		'val':(puts_val,contact_val),
		'test':(puts_test,contact_test)
		}

def create_MPL_backwards(conf,config_wandb=None):
	"""
	Creates a Keras model that takes as input the mechanical response and 
	gives as output the contact parameters.
	"""
	#inputs_outputs,contact_params=load_data_conf(conf)
	inputs_outputs,contact_params=load_data()
	splits=split_data(inputs_outputs,contact_params)
	num_samples, sequence_length, num_features = np.shape(splits['train'][0]) #outputs
	num_samples2, num_labels = np.shape(splits['train'][1]) #contact params
	
	#model definition
	n_neurons=100
	model = keras.Sequential()
	model.add(keras.layers.Input(shape=(sequence_length,num_features)))
	model.add(keras.layers.Flatten(input_shape=(None,sequence_length,n_neurons)))
	model.add(keras.layers.Normalization(axis=-1, mean=None, variance=None))
	model.add(keras.layers.Dense(n_neurons*3,activation='sigmoid'))
	model.add(keras.layers.Dense(n_neurons,activation='relu'))
	model.add(keras.layers.Dense(num_labels, name='predicted_contact_params'))

	model.summary()
	model.compile(optimizer='rmsprop',loss='mse', metrics=['mae'])

	#Training
	early_stopping_monitor = EarlyStopping(patience=10)
	model.fit(splits['train'][0],splits['train'][1],
			   batch_size=32,
               epochs=40,
               validation_data=(splits['val'][0],splits['val'][1]),
               callbacks=[early_stopping_monitor])

	#Testing
	testing_model(splits['test'][0][0:25],splits['test'][1][0:25],model,'MPL')

def testing_model(outputs, contact_params, model, name_model):
	"""
	Calculates the contact_params predicted by the model for a given output
	"""
	titles_labels=['E','$\\nu$','kr','$\eta$','$\mu$','$e_0$']
	fig, ax = plt.subplots(2, 3)
	contact_params_predicted=[]
	contact_params_predicted=model.predict(outputs)
	for i,prediction_i in enumerate(contact_params_predicted):
		for j in range(len(prediction_i)):
			x = j % 2
			y = j // 2
			ax[x,y].plot(i,prediction_i[j],'r.') #prediction
			ax[x,y].plot(i,contact_params[i][j],'b.') #truth
			ax[x,y].set_title(titles_labels[j])
	ax[1,2].plot(0,0,'r.',label='prediction')
	ax[1,2].plot(0,0,'b.',label='truth')
	ax[1,2].legend()
	plt.tight_layout()
	fig.savefig(name_model+'.pdf')

def rnn(conf,config_wandb=None):
	"""
	Creates an RNN model using Keras LSTM layer.
	Problem of type from multiple (sequence: outputs) to one (vector: contact_params)
	Params: 
		conf: the confinment stress
		config_wandb: dictionary with model variables. Useful when tracking with wandb
	"""
	#inputs_outputs,contact_params=load_data_conf(conf)
	splits=split_data(inputs_outputs,contact_params)
	num_samples, sequence_length, num_features = np.shape(splits['train'][0]) #outputs
	num_samples2, num_labels = np.shape(splits['train'][1]) #contact params

	#model definition
	units_LSTM=100 ; units_dense=100
	normalization_layer=False
	activation_LSTM='tanh'; activation_Dense='relu'

	if not config_wandb==None:
		units_LSTM =config_wandb["units_LSTM"] 
		units_dense=config_wandb["units_dense"]
		activation_LSTM=config_wandb["activation_LSTM"]
		activation_Dense=config_wandb["activation_Dense"]
		normalization_layer=config_wandb["normalization_layer"]

	model_lstm = keras.Sequential()
	model_lstm.add(keras.layers.Input(shape=(sequence_length,num_features)))
	if normalization_layer: model_lstm.add(keras.layers.Normalization(axis=-1, mean=None, variance=None))
	model_lstm.add(keras.layers.LSTM(units_LSTM,activation=activation_LSTM))
	model_lstm.add(keras.layers.Dense(units_dense,activation=activation_Dense))
	model_lstm.add(keras.layers.Dense(num_labels, name='predicted_contact_params'))

	model_lstm.summary()
	if config_wandb==None:	
		model_lstm.compile(optimizer='adam',loss='mse', metrics=['mae'])

		#Training
		early_stopping_monitor = EarlyStopping(patience=10,restore_best_weights=True)
		model_lstm.fit(splits['train'][0],splits['train'][1],
				   batch_size=32,
	               epochs=10,
	               validation_data=(splits['val'][0],splits['val'][1]),
	               callbacks=[early_stopping_monitor])
	else:
		model_lstm.compile(optimizer=config_wandb["optimizer"],
							loss=config_wandb["loss"], metrics=['mae'])

		#Training
		early_stopping_monitor = EarlyStopping(patience=config_wandb["patience"],
												restore_best_weights=True)
		wandb_callback = wandb.keras.WandbCallback(
            monitor='val_root_mean_squared_error', 
            save_model=(True))	
		model_lstm.fit(splits['train'][0],splits['train'][1],
				   batch_size=config_wandb["batch_size"],
	               epochs=config_wandb["epochs"],
	               validation_data=(splits['val'][0],splits['val'][1]),
	               callbacks=[early_stopping_monitor,wandb_callback])

		#TODO: Try this with pandas
		#wandb.log({"pr": wandb.plot.pr_curve(splits['test'][2][0:10], model_lstm.predict(splits['test'][1][0:10]))})

	#Testing
	testing_model(splits['test'][0][0:10],splits['test'][1][0:10],model_lstm,'LSTM')

def main_local(conf):
	tf.keras.backend.clear_session()
	# MPL model
	create_MPL_backwards(conf)

	# RNN model
	#rnn(conf)

def experiment_tracking_wandb(conf):
	
	wandb.init(project="RNN_Luisa_grainLearning", entity="luisaforozco",
		config = {
		  "learning_rate": 0.001,
		  "epochs": 15,
		  "batch_size": 50,
		  "optimizer": 'rmsprop',
		  "loss": 'mse',
		  "patience": 10,
		  "architecture": 'RNN-LSTM',
		  "units_LSTM": 100,
		  "units_dense": 100,
		  "activation_LSTM": 'tanh', #must me tanh otherwise 'Layer lstm will not use cuDNN kernels since it doesn't meet the criteria' and runs too slow
		  "activation_Dense": 'relu',
		  "normalization_layer": False
		})

	tf.keras.backend.clear_session()
	rnn(conf, config_wandb=wandb.config)

	wandb.finish()

if __name__ == '__main__':
	main_local()
	#experiment_tracking_wandb(conf)

	