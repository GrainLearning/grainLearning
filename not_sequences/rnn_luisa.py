import os, h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

import wandb
import IPython

"-------- DATA Pre-Processing ---------"
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
		outputs_i=[]
		for i in range(0,len(f[k]['drained']['outputs'])): #adding the confinment to the outputs
			output_i = f[k]['drained']['outputs'][i]
			conf_array=np.repeat(np.array(float(k)),np.shape(output_i)[0])
			outputs_i.append(np.column_stack((output_i,conf_array)))
		outputs.extend(np.concatenate((f[k]['drained']['inputs'],outputs_i),axis=2))
		contact_params.extend(f[k]['drained']['contact_params'])

	if(np.isnan(np.array(contact_params)).any()): print("Nans in contact_params")
	if(np.isnan(np.array(outputs)).any()): print("Nans in inputs_outputs")
	contact_params=standardize_labels(np.array(contact_params))
	outputs=standardize_features(np.array(outputs))
	
	return np.asarray(outputs),np.asarray(contact_params)

def standardize_labels(contact_params):
	"""
	Normalize the labels (contact_params) as: x-mu(x) /sigma(x)
	"""
	means = np.mean(contact_params,axis=0)
	stds  = np.std(contact_params,axis=0)
	for i in range(0,np.shape(contact_params)[1]):
		contact_params[:,i]=(contact_params[:,i]-means[i])/stds[i]
	return contact_params

def standardize_features(inputs_outputs):
	"""
	Normalize the features (inputs and outputs) as: x-mu(x) /sigma(x)
	"""
	means = np.mean(inputs_outputs,axis=(0,1))
	stds  = np.std(inputs_outputs,axis=(0,1))
	for i in range(0,np.shape(inputs_outputs)[2]): #This should be ok, I've checked
		inputs_outputs[:,:,i]=(inputs_outputs[:,:,i]-means[i])/stds[i]
	return inputs_outputs

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

"----------- MODELS -----------"
def mlp(config_wandb=None):
	"""
	Creates a Keras model that takes as input the mechanical response and 
	gives as output the contact parameters.
	The model is a sequential multi layer perceptron.
	"""
	inputs_outputs,contact_params=load_data()
	splits=split_data(inputs_outputs,contact_params)
	num_samples, sequence_length, num_features = np.shape(splits['train'][0]) #outputs
	num_samples2, num_labels = np.shape(splits['train'][1]) #contact params
	
	#model definition
	n_neurons=100
	model = keras.Sequential()
	model.add(keras.layers.Input(shape=(sequence_length,num_features)))
	model.add(keras.layers.Flatten(input_shape=(None,sequence_length,n_neurons)))
	#model.add(keras.layers.Normalization(axis=-1, mean=None, variance=None)) #not necessary if standardize_labels
	model.add(keras.layers.Dense(n_neurons*3,activation='sigmoid'))
	model.add(keras.layers.Dense(n_neurons,activation='relu'))
	model.add(keras.layers.Dense(num_labels, name='predicted_contact_params'))

	model.summary()
	model.compile(optimizer='rmsprop',loss='mse', metrics=['mae',tf.keras.metrics.MeanAbsolutePercentageError()])

	#Training
	early_stopping_monitor = EarlyStopping(patience=10)
	model.fit(splits['train'][0],splits['train'][1],
			   batch_size=32,
               epochs=40,
               validation_data=splits['val'],
               callbacks=[early_stopping_monitor])

	#Testing
	testing_model(splits['test'][0][0:25],splits['test'][1][0:25],model,'MPL')

def rnn(config_wandb=None):
	"""
	Creates an RNN model using Keras LSTM layer.
	Problem of type from multiple (sequence: outputs) to one (vector: contact_params)
	Params: 
		config_wandb: dictionary with model variables. Useful when tracking with wandb
	"""
	inputs_outputs,contact_params=load_data()
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
	               validation_data=(splits['val']),
	               callbacks=[early_stopping_monitor,wandb_callback])

		#TODO: Try this with pandas
		#wandb.log({"pr": wandb.plot.pr_curve(splits['test'][2][0:10], model_lstm.predict(splits['test'][1][0:10]))})

	#Testing
	testing_model(splits['test'][0][0:20],splits['test'][1][0:20],model_lstm,'LSTM')

def cnn(config_wandb=None):
	inputs_outputs,contact_params=load_data()
	splits=split_data(inputs_outputs,contact_params)
	num_samples, sequence_length, num_features = np.shape(splits['train'][0]) #outputs
	num_samples2, num_labels = np.shape(splits['train'][1]) #contact params
	
	units_cnn=100
	units_dense=200
	if not config_wandb==None: 
		units_cnn = config_wandb["units_CNN"] 
		units_dense=config_wandb["units_dense"] 

	#model that works well including e_0
	model_cnn = keras.Sequential([
		keras.layers.Input(shape=(sequence_length,num_features)),
		keras.layers.Conv1D(units_cnn,2,activation='relu',padding="same"),
		keras.layers.Conv1D(units_cnn*2,2,activation='relu',padding="same"),
		keras.layers.AveragePooling1D(pool_size=2),
		keras.layers.Conv1D(units_cnn*2,2,activation='relu',padding="same"),
		keras.layers.MaxPooling1D(pool_size=2),
		keras.layers.Conv1D(units_cnn*2,2,activation='relu',padding="same"),
		keras.layers.MaxPooling1D(pool_size=2),
		keras.layers.Conv1D(units_cnn*2,2,activation='relu',padding="same"),
		#keras.layers.MaxPooling1D(pool_size=2),
		#keras.layers.Flatten(),
		keras.layers.GlobalMaxPooling1D(),
		keras.layers.Dense(units_dense,activation='relu'),
		keras.layers.Dense(num_labels,name='predicted_contact_params')
		])

	'''model_cnn = keras.Sequential([
		keras.layers.Input(shape=(sequence_length,num_features)),
		keras.layers.Conv1D(units_cnn,2,activation='relu',padding="same"),
		#keras.layers.Conv1D(units_cnn*2,2,activation='relu',padding="same"),
		keras.layers.AveragePooling1D(pool_size=2),
		keras.layers.Conv1D(units_cnn*2,4,activation='relu',padding="same"),
		keras.layers.MaxPooling1D(pool_size=2),
		keras.layers.Conv1D(units_cnn*2,8,activation='relu',padding="same"),
		#keras.layers.MaxPooling1D(pool_size=2),
		#keras.layers.Conv1D(units_cnn*2,2,activation='relu',padding="same"),
		#keras.layers.MaxPooling1D(pool_size=2),
		#keras.layers.Flatten(),
		keras.layers.GlobalMaxPooling1D(),
		keras.layers.Dense(units_dense,activation='relu'),
		keras.layers.Dense(num_labels,name='predicted_contact_params')
		])'''
	
	if config_wandb==None:
		model_cnn.compile(optimizer='adam',loss='mse',
					  metrics=['mae',tf.keras.metrics.MeanAbsolutePercentageError()])
		#Training
		early_stopping_monitor = EarlyStopping(patience=10,restore_best_weights=True)
		training_history=model_cnn.fit(splits['train'][0],splits['train'][1],
			   batch_size=32,
               epochs=30,
               validation_data=splits['val'],
               callbacks=[early_stopping_monitor])
	else:

		opt_adam=tf.keras.optimizers.Adam(learning_rate=config_wandb["learning_rate"])
		model_cnn.compile(optimizer=opt_adam,loss=config_wandb["loss"],
					  metrics=['mae',tf.keras.metrics.MeanAbsolutePercentageError()])
		#Training
		early_stopping_monitor = EarlyStopping(patience=config_wandb["patience"],
												restore_best_weights=True)
		wandb_callback = wandb.keras.WandbCallback(
            monitor='val_root_mean_squared_error', 
            save_model=(True))	
		training_history=model_cnn.fit(splits['train'][0],splits['train'][1],
				   batch_size=config_wandb["batch_size"],
	               epochs=config_wandb["epochs"],
	               validation_data=splits['val'],
	               callbacks=[early_stopping_monitor,wandb_callback])
	
	best_score_train_set=min(training_history.history["loss"])
	best_score_val_set=min(training_history.history["val_loss"])
	print(f'Best epoch train loss: {best_score_train_set}')
	print(f'Best epoch val loss: {best_score_val_set}')

	#Testing
	testing_model(splits['test'][0][0:20],splits['test'][1][0:20],model_cnn,'CNN')

def wavenet():
	inputs_outputs,contact_params=load_data()
	splits=split_data(inputs_outputs,contact_params)
	num_samples, sequence_length, num_features = np.shape(splits['train'][0]) #outputs
	num_samples2, num_labels = np.shape(splits['train'][1]) #contact params

	model_wavenet = keras.models.Sequential()
	model_wavenet.add(keras.layers.Input(shape=(sequence_length,num_features)))
	for rate in (1, 2, 4, 8) * 2:
   	     model_wavenet.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal",
                                  activation="relu", dilation_rate=rate))
	model_wavenet.add(keras.layers.Conv1D(filters=10, kernel_size=1))
	model_wavenet.add(keras.layers.Flatten())
	model_wavenet.add(keras.layers.Dense(num_labels, name='predicted_contact_params'))
	
	model_wavenet.compile(loss="mse", optimizer="adam", metrics=['mae',tf.keras.metrics.MeanAbsolutePercentageError()])

	#Trainning
	early_stopping_monitor = EarlyStopping(patience=10,restore_best_weights=True)
	training_history = model_wavenet.fit(splits['train'][0],splits['train'][1],batch_size=32,
		epochs=20,validation_data=splits['val'],callbacks=[early_stopping_monitor])

	#Testing
	testing_model(splits['test'][0][0:20],splits['test'][1][0:20],model_wavenet,'Wavenet')

"----------- Model EVALUATION -----------"
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
			#ax[x,y].plot(i,prediction_i[j],'r.',fillstyle='none') #prediction
			#ax[x,y].plot(i,contact_params[i][j],'b.',fillstyle='none') #truth
			ax[x,y].plot(contact_params[i][j],prediction_i[j],'k.')
			ax[x,y].set_title(titles_labels[j])
			ax[x,y].set_xlabel('truth');ax[x,y].set_ylabel('prediction')
	#ax[1,2].plot(np.nan,np.nan,'r.',fillstyle='none',label='prediction')
	#ax[1,2].plot(np.nan,np.nan,'b.',fillstyle='none',label='truth')
	#ax[1,2].legend()
	plt.tight_layout()
	fig.savefig(name_model+'.pdf')

"----------- MAIN -----------"
def run_local():
	tf.keras.backend.clear_session()
	# MLP model
	#mlp()

	# RNN model
	#rnn()

	#Convolutional 1D
	cnn()

	#wavenet
	#wavenet()

def experiment_tracking_wandb():
	"""
	Initializing wandb and config dictionary.
	At the moment only implemented for tracking experiments of the RNN model.
	"""
	
	config_rnn={
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
		}

	config_cnn={
		  "learning_rate": 1E-4,
		  "epochs": 150,
		  "batch_size": 50,
		  "optimizer": 'adam',
		  "loss": 'mse',
		  "patience": 50,
		  "architecture": '5 conv1D, 3 pool, GlobalMax, 2 dense',
		  "units_CNN": 100,
		  "units_dense": 200
		}

	wandb.init(project="CNN_grainLearning", entity="luisaforozco",
		config = config_cnn)

	tf.keras.backend.clear_session()

	#rnn(config_wandb=wandb.config)
	cnn(config_wandb=wandb.config)

	wandb.finish()

if __name__ == '__main__':
	run_local()
	#experiment_tracking_wandb()
	