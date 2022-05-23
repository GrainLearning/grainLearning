import h5py
import numpy as np
import os

DATA_DIR = '/Users/aronjansen/Documents/grainsData/graph_test_data/'
FILENAME = 'simState_drained_200_10000_0_gnn_data.hdf5'
SAVE_DIR = '/Users/aronjansen/Documents/grainsData/graph_test_data/graphs.hdf5'
SAVE_DIR_SHORT = '/Users/aronjansen/Documents/grainsData/graph_test_data/graphs_short.hdf5'

file_names = [fn for fn in os.listdir(DATA_DIR) if fn.endswith('_gnn_data.hdf5')]

interactions = []
for fname in file_names:
    f = h5py.File(DATA_DIR + fname, 'r')
    f = f['0.1e6']['drained']['10000']
    num_interactions = f['outputs_inters'][:][0].shape[1]
    interactions.append(num_interactions)
interactions = np.array(interactions)

max_interactions = np.max(interactions)

sources, destinations, edge_features, node_features = [], [], [], []
input_features, contact_params = [], []

def write_to_file(num_steps, filename):
    if os.path.exists(filename):
        os.remove(filename)
    f_perstep = h5py.File(filename, 'a')
    for step in range(0, num_steps):
        fname = f'simState_drained_200_10000_{step}_gnn_data.hdf5'
        f = h5py.File(DATA_DIR + fname, 'r')
        f = f['0.1e6']['drained']['10000']
        interactions_raw = f['outputs_inters'][:][0]
        src = interactions_raw[0, :].astype(int)
        dst = interactions_raw[1, :].astype(int)
        e = interactions_raw[2:]
        e = np.transpose(e, (1, 0))
        n = np.transpose(f['outputs_bodies'][:][0], (1, 0))

        if step == 0:
            f_perstep['contact_params'] = contact_params

        f_step = f_perstep.require_group(f'{step}')
        f_step['sources'] = src
        f_step['destinations'] = dst
        f_step['edge_features'] = e
        f_step['node_features'] = n
        f_step['input_features'] = f['inputs'][:][0]

write_to_file(num_steps=201, filename=SAVE_DIR)
write_to_file(num_steps=5, filename=SAVE_DIR_SHORT)
