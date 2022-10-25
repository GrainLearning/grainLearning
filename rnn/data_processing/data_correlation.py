import os, h5py
import numpy as np
import pandas as pd
import IPython
import matplotlib.pyplot as plt
import seaborn as sn

def load_data(path_to_data):
	"""
	Loads the data from the hdf5 file consolidated by Aron.
	Calculates the following macroscopic observations: void ratio e, p and q at peak and residual.
	Params: path_to_data is a string with the path to the hdf5 file.
	Returns a matrix containing the contact parameters and the macroscopic observations.
	The variables are in each one of the columns and each row is an observation (DEM simulation).
	"""

	f = h5py.File(path_to_data, 'r') #binary file containing the consolidating data by Aron
	contact_params,outputs = ([] for i in range(0,2))

	for k in f.keys(): #Adding info a different confinment pressures to the same list
		num_samples, num_features = np.shape(f[k]['drained']['contact_params'])
		for i in range(0,len(f[k]['drained']['outputs'])):
			output_i = f[k]['drained']['outputs'][i]
			e_peak_i = np.max(output_i[0])
			p_peak_i = np.max(output_i[1])
			q_peak_i = np.max(output_i[2])
			e_residual_i = np.mean(output_i[0][-10:])
			p_residual_i = np.mean(output_i[1][-10:])
			q_residual_i = np.mean(output_i[2][-10:])
			outputs.append([e_peak_i,p_peak_i,q_peak_i,e_residual_i,p_residual_i,q_residual_i])
		contact_params.extend(f[k]['drained']['contact_params'])
	matrix=np.concatenate((contact_params,outputs),axis=1)
	return matrix

def process_correlation_numpy(matrix):
	"""
	Calculates the correlation matrix using numpy and plots it.
	Params: matrix is a numpy array containing the variables (in columns) 
			and observations in the rows.
	"""
	covariance_matrix = np.corrcoef(matrix,rowvar=False)
	labels=["E","v","kr","eta","mu","e peak","p peak","q peak","e res","p res","q res"]
	fig, ax = plt.subplots()
	im = ax.imshow(covariance_matrix)
	im.set_clim(-1, 1)
	ax.grid(False)
	ax.xaxis.set(ticks=(0,1,2,3,4,5,6,7,8,9,10), ticklabels=(labels))
	ax.yaxis.set(ticks=(0,1,2,3,4,5,6,7,8,9,10), ticklabels=(labels))
	cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
	cbar.ax.set_ylabel('correlation coeff', rotation=270)
	plt.tight_layout()
	fig.savefig('correlation_matrix_numpy.pdf')
	plt.close()
	plt.clf()

def process_correlation_pandas(matrix):
	"""
	Calculates the correlation matrix using pandas and plots it.
	spearman is chosen because in this method there is no hypothesis saying that the data correlates linearly.
	Params: matrix is a numpy array containing the variables (in columns) 
			and observations in the rows.
	"""
	labels=['E','$\\nu$','kr','$\eta$','$\mu$',"e peak","p peak","q peak","e res","p res","q res"]
	df = pd.DataFrame(matrix, columns=["E","v","kr","eta","mu","e peak","p peak","q peak","e res","p res","q res"])

	corrMatrix=df.corr(method='spearman')
	maskd = np.triu(np.ones_like(corrMatrix, dtype=bool))
	sn.heatmap(corrMatrix,annot=True,annot_kws={"size": 6},mask=maskd,xticklabels=labels,yticklabels=labels)
	plt.tight_layout()
	plt.savefig('correlation_matrix_pandas.pdf')

if __name__ == '__main__':
	PATH_TO_DATA='./rnn_data.hdf5'
	matrix=load_data(PATH_TO_DATA)
	process_correlation_numpy(matrix)
	process_correlation_pandas(matrix)