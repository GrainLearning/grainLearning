#!/usr/bin/env python3
import sys
import numpy as np

def write_dict_to_file(data, file_name):
    """
    write a python dictionary data into a text file 
    """
    with open (file_name,'w') as f: 
        keys = data.keys()
        f.write('# ' + ' '.join(keys) + '\n')
        num = len(data[list(keys)[0]])
        for i in range(num):
            f.write(' '.join([str(data[key][i]) for key in keys]) + '\n')

a = float(sys.argv[1])
b = float(sys.argv[2])
description = sys.argv[3]

x_obs = np.arange(100)
y_sim = a * x_obs + b

# # write sim data in the .npy format
# data = {}
# data['a'] = a
# data['b'] = b
# data['f'] = y_sim
# data_file_name = 'linear_'+ description + '.npy'
# np.save(data_file_name,data)

# write sim data and parameter in text files 
data_file_name = 'linear_'+ description + '_sim.txt'
sim_data = {'f': y_sim}
write_dict_to_file(sim_data, data_file_name)

data_param_name = 'linear_'+ description + '_param.txt'
param_data = {'a': [a], 'b': [b]}
write_dict_to_file(param_data, data_param_name)
