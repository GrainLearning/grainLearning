"""
Preprocess data so that it can be used as input in an RNN.
Data consists of:
- contact_params (samples, 5)
- input_params (samples, sequence_length, x)
- output_params (samples, sequence_length, y)
"""
import numpy as np
import os
import h5py

contact_keys = [
        'E', # young's modulus  = 10^E
        'v', # poisson's ratio
        'mu', # sliding friction
        ]

input_keys = [
        'e',  # initial void ratio
        'conf',  # confining pressure (stored as group name already)
        'num',  # number of particles
]

output_keys = [
        ## particle shape info (in b.shape)
        [
        'shape.radius',
        ],
        ## particle motion info (in b.state)
        [
        'state.vel',
        'state.mass',
        'state.angVel',
        'state.angMom',
        'state.inertia',
        'state.refPos',
        'state.refOri',
        ],
        ## interaction connectivity info (in inter)
        [
        'id1',
        'id2',
        ## interaction geometry info (in inter.geom)
        'geom.penetrationDepth',
        'geom.shearInc',  # shear increment between particles
        'geom.normal',  # contact normal
        'geom.contactPoint',
        'geom.refR1',
        'geom.refR2',
        ## interaction physics info (in inter.phys)
        'phys.shearElastic',  # elastic component of the shear (tangential) force
        'phys.usElastic',  # elastic component of the shear displacement
        'phys.usTotal',  # total shear displacement
        'phys.ks',  # tangential stiffness
        'phys.shearForce',  # shear foce
        'phys.kn',  # normal stiffness
        'phys.normalForce',  # normal force
        ],
]

unused_keys_sequence = [
]

unused_keys_constant = [
]

SEQUENCE_LENGTH = 1
TARGET_DIR = '/home/cheng/DataGen/'
DATA_DIR = '/home/cheng/DataGen/'


def main(pressure, experiment_type,numParticles):
    data_dir = DATA_DIR + f'{pressure}/{experiment_type}/{numParticles}/'
    if not os.listdir(data_dir):
        print(f"Directory {data_dir} is empty.")
        return

    file_names = [fn for fn in os.listdir(data_dir) if fn.endswith('_0.yade.gz')]
    file_names = file_names[:2]
    print(file_names)

    contact_tensor = []
    inputs_tensor = []
    outputs_tensor = []
    for f in file_names:
        try:
            O.load(data_dir + f); O.step()
        except IOError:
            print('IOError', f, pressure)
            continue
        ### contact parameters        
        contact_tensor.append([
            O.materials[0].young,
            O.materials[0].poisson,
            O.materials[0].frictionAngle
            ])
        ### input data
        inputs_tensor.append([
            porosity()/(1-porosity()),
            getStress().trace()/3,
            len(O.bodies)])
        ### output data
        ## particle shape info
        for shapeKey in output_keys[0]:
            outputs_tensor.append([np.array(eval('b.'+shapeKey)) for b in O.bodies])
            print(outputs_tensor[-1][0])
        ## particle motion info
        for motionKey in output_keys[1]:
            outputs_tensor.append([np.array(eval('b.'+motionKey)) for b in O.bodies])
            print(outputs_tensor[-1][0])
        ## interaction info
        for interKey in output_keys[2]:
            outputs_tensor.append([np.array(eval('i.'+interKey)) for i in O.interactions])
            print(outputs_tensor[-1][0])

    # ~ # keras requires (batch, sequence_length, features) shape, so transpose axes (TODO)
    # ~ inputs_tensor = np.transpose(inputs_tensor, (0, 2, 1))
    # ~ outputs_tensor = np.transpose(outputs_tensor, (0, 2, 1))

    # ~ print(f'Created tensor of {outputs_tensor.shape[0]} samples,')

    with h5py.File(TARGET_DIR + 'gnn_data.hdf5', 'a') as f:
        grp = f.require_group(f'{pressure}/{experiment_type}')
        grp['contact_params'] = contact_tensor
        grp['inputs'] = inputs_tensor
        grp['outputs'] = outputs_tensor

        f.attrs['inputs'] = input_keys
        f.attrs['outputs'] = output_keys
        f.attrs['contact_params'] = contact_keys
        f.attrs['unused_keys_sequence'] = unused_keys_sequence
        f.attrs['unused_keys_constant'] = unused_keys_constant
    print(f"Added data to {TARGET_DIR + 'gnn_data.h5py'}")

if __name__ == '__main__':
    for pressure in ['0.1e6']:
        for experiment_type in ['drained']:
            for numParticles in ['10000','15000']:
                main(pressure, experiment_type, numParticles)
