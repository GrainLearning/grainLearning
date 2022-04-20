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

output_keys_bodies = [
        ## particle info (in b.shape and b.state)
        'shape.radius',
        'state.vel[0]',
        'state.vel[1]',
        'state.vel[2]',
        'state.mass',
        'state.angVel[0]',
        'state.angVel[1]',
        'state.angVel[2]',
        'state.angMom[0]',
        'state.angMom[1]',
        'state.angMom[2]',
        'state.inertia[0]',
        'state.inertia[1]',
        'state.inertia[2]',
        'state.refPos[0]',
        'state.refPos[1]',
        'state.refPos[2]',
        'state.refOri[0]',
        'state.refOri[1]',
        'state.refOri[2]',
        'state.refOri[3]',
        ]
output_keys_inters = [
        ## interaction connectivity info (in inter)
        'id1',
        'id2',
        ## interaction geometry info (in inter.geom)
        'geom.penetrationDepth',
        'geom.shearInc[0]',  # shear increment x between particles
        'geom.shearInc[1]',  # shear increment y between particles
        'geom.shearInc[2]',  # shear increment z between particles
        'geom.normal[0]',  # contact normal x
        'geom.normal[1]',  # contact normal y
        'geom.normal[2]',  # contact normal z
        'geom.contactPoint[0]',
        'geom.contactPoint[1]',
        'geom.contactPoint[2]',
        'geom.refR1',
        'geom.refR2',
        ## interaction physics info (in inter.phys)
        'phys.shearElastic[0]',  # elastic component of the shear (tangential) force x
        'phys.shearElastic[1]',  # elastic component of the shear (tangential) force y
        'phys.shearElastic[2]',  # elastic component of the shear (tangential) force z
        'phys.usElastic[0]',  # elastic component of the shear displacement x
        'phys.usElastic[1]',  # elastic component of the shear displacement y
        'phys.usElastic[2]',  # elastic component of the shear displacement z
        'phys.usTotal[0]',  # total shear displacement x
        'phys.usTotal[1]',  # total shear displacement y
        'phys.usTotal[2]',  # total shear displacement z
        'phys.ks',  # tangential stiffness
        'phys.shearForce[0]',  # shear foce x
        'phys.shearForce[1]',  # shear foce y
        'phys.shearForce[2]',  # shear foce z
        'phys.kn',  # normal stiffness
        'phys.normalForce[0]',  # normal force x
        'phys.normalForce[1]',  # normal force y
        'phys.normalForce[2]',  # normal force z
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
    file_names = file_names[:1]
    print(file_names)

    contact_tensor = []
    inputs_tensor = []
    outputs_tensor_bodies = []
    outputs_tensor_inters = []
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
        ## particle info
        bodies_data = []
        for bodyKey in output_keys_bodies:
            bodies_data.append([float(eval('b.'+bodyKey)) for b in O.bodies])
        outputs_tensor_bodies.append(bodies_data)
        ## interaction info
        inters_data = []
        for interKey in output_keys_inters:
            inters_data.append([float(eval('i.'+interKey)) for i in O.interactions if i.isReal])
        outputs_tensor_inters.append(inters_data)

    # keras requires (batch, sequence_length, features) shape, so transpose axes (TODO, originally by Aron)
    numData = len(outputs_tensor_bodies)
    numNodes = len(O.bodies)
    numEdges = O.interactions.countReal()
    outputs_tensor_bodies = np.array(outputs_tensor_bodies)
    print(outputs_tensor_bodies.shape,numNodes)
    outputs_tensor_inters = np.array(outputs_tensor_inters)
    print(outputs_tensor_inters.shape,numEdges)

    print(f'Created tensor of {len(outputs_tensor_bodies)} samples,')
    print(f'Created tensor of {len(outputs_tensor_inters)} samples,')
    print(f'Created tensor of {len(contact_tensor)} samples,')
    print(f'Created tensor of {len(inputs_tensor)} samples,')

    with h5py.File(TARGET_DIR + 'tmp/gnn_data.hdf5', 'a') as f:
        grp = f.require_group(f'{pressure}/{experiment_type}/{numParticles}')
        grp['contact_params'] = contact_tensor
        grp['inputs'] = inputs_tensor
        grp['outputs_bodies'] = outputs_tensor_bodies
        grp['outputs_inters'] = outputs_tensor_inters

        f.attrs['contact_params'] = contact_keys
        f.attrs['inputs'] = input_keys
        f.attrs['outputs_bodies'] = output_keys_bodies
        f.attrs['outputs_inters'] = output_keys_inters
        f.attrs['unused_keys_sequence'] = unused_keys_sequence
        f.attrs['unused_keys_constant'] = unused_keys_constant
    print(f"Added data to {TARGET_DIR + 'gnn_data.h5py'}")

if __name__ == '__main__':
    for pressure in ['0.1e6']:
        for experiment_type in ['drained']:
            for numParticles in ['10000','15000']:
                main(pressure, experiment_type, numParticles)
