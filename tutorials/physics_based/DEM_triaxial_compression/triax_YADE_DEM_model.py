# encoding: utf-8

# default material parameters
readParamsFromTable(
    # no. of your simulation
    key=0,
    # Young's modulus
    E_m=8.8328129646e+00,
    # Poisson's ratio
    v=1.1768868293e-01,
    # rolling/bending stiffness
    kr=1.0968388788e-01,
    # rolling/bending plastic limit
    eta=7.3928627878e-01,
    # final friction coefficient
    mu=2.9648597170e+01,
    # number of particles
    num=1000,
    # initial confining pressure
    conf=0.5e6,
    unknownOk=True
)

import numpy as np
from yade.params import table
from yade import pack
from grainlearning.tools import get_keys_and_data, write_dict_to_file

# check if run in batch mode
isBatch = runningInBatch()
if isBatch:
    description = O.tags['description']
else:
    description = 'triax_DEM_test_run'

#: Simulation control parameters
num = table.num  # number of soil particles
dScaling = 1e3  # density scaling
e = 0.68  # initial void ratio
conf = table.conf  # confining pressure
rate = 0.1  # strain rate (decrease this for serious calculations)
damp = 0.2  # damping coefficient
stabilityRatio = 1.e-2  # threshold for quasi-static condition (decrease this for serious calculations)
stressTolRatio = 1.e-2  # tolerance for stress goal
initStabilityRatio = 1.e-2  # initial stability threshold
obsCtrl = 'e_z'  # key for simulation control
lowDamp = 0.2  # damping coefficient
highDamp = 0.9
debug = False

#: load strain/stress data for quasi-static loading
obs_file_name = "triax_data_DEM.dat"
obs_data = get_keys_and_data(obs_file_name)
obs_ctrl_data = list(0.01 * obs_data[obsCtrl])
obs_ctrl_data.reverse()

#: Soil sphere parameters
E = pow(10, table.E_m)  # micro Young's modulus
v = table.v  # micro Poisson's ratio
kr = table.kr  # rolling/bending stiffness
eta = table.eta  # rolling/bending plastic limit
mu = table.mu  # contact friction during shear
ctrMu = table.mu  # use small mu to prepare dense packing?
rho = 2650 * dScaling  # soil density

#: create materials
spMat = O.materials.append(
    CohFrictMat(young=E, poisson=v, frictionAngle=radians(ctrMu), density=rho, isCohesive=False,
                alphaKr=kr, alphaKtw=kr, momentRotationLaw=True, etaRoll=eta, etaTwist=eta))

# create dictionary to store simulation data
sim_data = {}
sim_data['e_z'] = []
sim_data['e_v'] = []
sim_data['s33_over_s11'] = []

#: define the periodic box where a certain configuration of particles is loaded
O.periodic = True
sp = pack.SpherePack()
cfg_file = 'PeriSp_' + str(num) + f'_{e}.txt'
with open(cfg_file, 'r') as f:
    first_line = f.readline()
    sizes = first_line.split()[1:]
    sizes = [float(s) for s in sizes]
O.cell.hSize = Matrix3(sizes[0], 0, 0, 0, sizes[1], 0, 0, 0, sizes[1])
sp.load(cfg_file)

# load spheres to simulation
spIds = sp.toSimulation(material=spMat)

# define engines
O.engines = [
    ForceResetter(),
    InsertionSortCollider([Bo1_Sphere_Aabb()]),
    InteractionLoop(
        [Ig2_Sphere_Sphere_ScGeom6D()],
        [Ip2_CohFrictMat_CohFrictMat_CohFrictPhys()],
        [Law2_ScGeom6D_CohFrictPhys_CohesionMoment(
            always_use_moment_law=True,
            useIncrementalForm=True
        )],
    ),
    GlobalStiffnessTimeStepper(timestepSafetyCoefficient=0.8),
    PeriTriaxController(label='triax',
                        # whether they are strains or stresses
                        stressMask=7,
                        # confining stress
                        goal=(-conf, -conf, -conf),
                        # strain rate
                        maxStrainRate=(10. * rate, 10. * rate, 10. * rate),
                        # type of servo-control
                        dynCell=True,
                        # wait until the unbalanced force goes below this value
                        maxUnbalanced=stabilityRatio,
                        # turn on checkVoidRatio after finishing initial compression
                        relStressTol=stressTolRatio,
                        # function to call after the goal is reached
                        doneHook='compactionFinished()',
                        ),
    NewtonIntegrator(damping=damp, label='newton'),
]


def compactionFinished():
    if unbalancedForce() < stabilityRatio:
        if debug: print('compaction finished')
        # set the current cell configuration to be the reference one
        O.cell.trsf = Matrix3.Identity
        O.cell.velGrad = Matrix3.Zero
        setRefSe3()
        # set loading type: constant pressure in x,y, 8.5% compression in z
        triax.stressMask = 3;
        triax.globUpdate = 1;
        # allow faster deformation along x,y to better maintain stresses
        triax.maxStrainRate = (10 * rate, 10 * rate, rate)
        # set damping to a normal value
        newton.damping = lowDamp
        print('start trixial shearing.')
        startLoading()


# start to load the packing
def startLoading():
    global obs_ctrl_data
    if debug: print('startLoading')
    strain = obs_ctrl_data.pop()
    triax.goal = Vector3(-conf, -conf, -strain)
    triax.maxUnbalanced = initStabilityRatio
    triax.relStressTol = stressTolRatio
    triax.doneHook = 'calmSystem()'
    newton.damping = lowDamp


# switch on high background damping to stablize the packing
def calmSystem():
    if debug: print('calmSystem')
    newton.damping = highDamp
    triax.maxUnbalanced = stabilityRatio
    triax.doneHook = 'addPlotData()'


def addPlotData():
    global obs_ctrl_data
    if debug: print('addPlotData')
    s = triax.stress
    s33_over_s11 = 2. * s[2] / (s[0] + s[1])
    e_x, e_y, e_z = -triax.strain
    e_v = e_x + e_y + e_z
    n = porosity()
    print("Triax goal is: ", triax.goal, "dt=", O.dt, "numIter=", O.iter, "real time=", O.realtime)  # e = n/(1.-n)
    sim_data['e_z'].append(100 * e_z)
    sim_data['e_v'].append(100 * e_v)
    sim_data['s33_over_s11'].append(s33_over_s11)
    if len(obs_ctrl_data) != 0:
        startLoading()
    else:
        # write simulation data into a text file
        data_file_name = f'{description}_sim.txt'
        data_param_name = f'{description}_param.txt'
        # initialize data dictionary
        param_data = {}
        for name in table.__all__:
            param_data[name] = eval('table.' + name)
        # write simulation data into a text file
        write_dict_to_file(sim_data, data_file_name)
        write_dict_to_file(param_data, data_param_name)
        O.pause()


# run in batch mode
t0 = O.realtime
O.run()
waitIfBatch()
