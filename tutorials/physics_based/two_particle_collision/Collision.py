# encoding: utf-8

# default material parameters
readParamsFromTable(
    # no. of your simulation
    key=0,
    # Density
    rho=2450,
    # Exponent of Young's modulus
    E_m=9.3,
    # Poisson's ratio
    nu=0.2,
    # final friction coefficient
    mu=0.4,
    # timestepSafetyCoefficient
    safe=0.1,
)

# import modules
import numpy as np

from yade.params import table
from yade import plot

from grainlearning.tools import write_dict_to_file

# check if run in batch mode
isBatch = runningInBatch()
if isBatch:
    description = O.tags['description']
else:
    description = 'collision_test_run'

# glass bead parameters (units: ug->1e-9kg; mm->1e-3m; ms->1e-3s)
lenScale = 1e3  # length in mm <- 1e-3 m
sigScale = 1  # Stress in ug/(mm*ms^2) <- Pa
rhoScale = 1  # Density in ug/mm^3 <- kg/m^3


# function to save simulation data and stop simulation
def add_sim_data():
    # get interaction force between particle 1 and particle 2
    inter = O.interactions[0, 1]
    # get penetration depth
    u = inter.geom.penetrationDepth
    plot.addData(u=u, f=inter.phys.normalForce.norm())
    # move particle 1
    O.bodies[1].state.pos = O.bodies[1].state.refPos + Vector3(0, 0, -obs_ctrl_data.pop())
    if not obs_ctrl_data:
        data_file_name = f'{description}_sim.txt'
        data_param_name = f'{description}_param.txt'
        # initialize data dictionary
        param_data = {}
        for name in table.__all__:
            param_data[name] = eval('table.' + name)
        # write simulation data into a text file
        write_dict_to_file(plot.data, data_file_name)
        write_dict_to_file(param_data, data_param_name)
        O.pause()


obs_file = "collision_obs.txt"

# define a load sequence
obs_ctrl_data = np.linspace(0.002, 0.01, 81).tolist()
obs_ctrl_data.reverse()

# create dictionary to store simulation data
plot.plots={'u':'f'}

# create materials
O.materials.append(
    FrictMat(young=pow(10, table.E_m), poisson=table.nu, frictionAngle=atan(table.mu), density=table.rho))

# create two particles
O.bodies.append(sphere(Vector3(0, 0, 0), 1, material=0, fixed=True))
O.bodies.append(sphere(Vector3(0, 0, 2), 1, material=0, fixed=True))

# define engines
O.engines = [
    ForceResetter(),
    InsertionSortCollider([Bo1_Sphere_Aabb()]),
    InteractionLoop(
        [Ig2_Sphere_Sphere_ScGeom()],
        [Ip2_FrictMat_FrictMat_MindlinPhys()],
        [Law2_ScGeom_MindlinPhys_Mindlin()]
    ),
    NewtonIntegrator(damping=0.0, label='newton'),
    # needs to add module collision before function name
    PyRunner(command='add_sim_data()', iterPeriod=1)
]

# set initial timestep
O.dt = table.safe * PWaveTimeStep()
# move particle 1
O.bodies[1].state.pos = O.bodies[1].state.refPos+ Vector3(0, 0, -obs_ctrl_data.pop())

# run DEM simulation
O.run()
waitIfBatch()
