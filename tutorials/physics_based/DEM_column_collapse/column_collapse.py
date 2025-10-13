# encoding: utf-8
# default material parameters
readParamsFromTable(
    # no. of your simulation
    key=0,
    # Young's modulus
    E_m=9,
    # Poisson's ratio
    v=0.1,
    # rolling/bending stiffness
    kr=0.1,
    # rolling/bending plastic limit
    eta=0.7,
    # initial friction coefficient
    ctrMu=10,
    # final friction coefficient
    mu=30,
    # wall friction coefficient
    wallMu=60,    
    # number of particles
    num=1000,
    unknownOk=True
)

import numpy as np
from yade.params import table
from yade import pack, plot
from yade.export import VTKExporter
import sys
sys.path.append('/home/jovyan/GL/GrainLearning')
from grainlearning.tools import get_keys_and_data, write_dict_to_file

# check if run in batch mode
isBatch = runningInBatch()
if isBatch:
    description = O.tags['description']
else:
    description = 'column_collapse_DEM_test_run'

# Domain size
width = 0.7
height = 4
depth = 0.7

#: Simulation control parameters
num = table.num  # number of soil particles
e = 0.5  # initial void ratio
damp = 0.2  # damping coefficient
stabilityRatio = 1.e-3  # threshold for quasi-static condition (decrease this for serious calculations)
debug = False

#: Soil sphere parameters
E = pow(10, table.E_m)  # micro Young's modulus
v = table.v  # micro Poisson's ratio
kr = table.kr  # rolling/bending stiffness
eta = table.eta  # rolling/bending plastic limit
mu = radians(table.mu)  # contact friction during shear
ctrMu = radians(table.ctrMu)  # use small mu to prepare dense packing?
wallMu = radians(table.wallMu)  # wall friction
rho = 2650  # soil density
create_packing = False  # create a new packing or load an existing one?

#: create materials
spMat = O.materials.append(
    CohFrictMat(young=E, poisson=v, frictionAngle=ctrMu, density=rho, isCohesive=False,
                alphaKr=kr, alphaKtw=kr, momentRotationLaw=True, etaRoll=eta, etaTwist=eta))

wallMat = O.materials.append(
    CohFrictMat(young=E, poisson=v, frictionAngle=ctrMu, density=rho, isCohesive=False,
                momentRotationLaw=False, label='walls'))

# create rectangular box from facets
mn, mx = Vector3(0, 0, 0), Vector3(width, height, depth)
# create walls
left_wall = O.bodies.append(wall((0,0,0), axis=0, sense=1, material=wallMat))
right_wall = O.bodies.append(wall((width,0,0), axis=0, sense=-1, material=wallMat))
bottom_wall = O.bodies.append(wall((0,0,0), axis=1, sense=1, material=wallMat))

# create empty sphere packing
sp = pack.SpherePack()
if create_packing:
    # generate randomly spheres with uniform radius distribution
    sp.makeCloud((mn[0], mn[1], (mn+mx)[2]), (mx[0], mx[1], (mn+mx)[2]), rMean=.01, rRelFuzz=.1, num=1000)
    # add the sphere pack to the simulation
    sp.toSimulation(material=spMat)
else:
    sp.load(f"initial_packing_{ctrMu:.3f}.txt")
    sp.toSimulation(material=spMat)

# make it quasi-2D by blocking motion in z direction
for b in O.bodies:
        if isinstance(b.shape, Sphere):
             b.state.blockedDOFs = 'zXY'  # make it quasi-2D

#: Define the engines
O.engines = [
        ForceResetter(),
        InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Facet_Aabb(), Bo1_Wall_Aabb()]),
        InteractionLoop(
                # handle sphere+sphere and facet+sphere collisions
                [Ig2_Sphere_Sphere_ScGeom6D(), Ig2_Facet_Sphere_ScGeom6D(), Ig2_Wall_Sphere_ScGeom()],
                [Ip2_CohFrictMat_CohFrictMat_CohFrictPhys()],
                [Law2_ScGeom6D_CohFrictPhys_CohesionMoment(
                        always_use_moment_law=True,
                        useIncrementalForm=True),
                Law2_ScGeom_FrictPhys_CundallStrack()],
        ),
        # GlobalStiffnessTimeStepper(timestepSafetyCoefficient=0.8),
        NewtonIntegrator(damping=damp, gravity=Vector3(0, -9.81, 0), label='newton'),
        PyRunner(command="check_unbalanced_before_collapse()",
                iterPeriod=1000,
                dead=False,
                firstIterRun=10000,
                label='check_unbalanced'),
        PyRunner(command="measure_run_out_distance()",
                iterPeriod=1000,
                dead=True,
                label='measure_run_out')
]

# default time step
O.dt = 0.5 * PWaveTimeStep()

# target number of measurement points to ensure consistent sequence length across runs
target_num_points = 200

def check_unbalanced_before_collapse():
    if unbalancedForce() < stabilityRatio:
        # Save the particle configuration before the collapse
        if not create_packing:
            sp.fromSimulation()
            sp.save(f"initial_packing_{ctrMu:.3f}.txt")
        # Remove the wall at x=max (right wall) to start the flow
        O.bodies.erase(right_wall)
        # Set higher friction for the rest of the simulation
        setContactFriction(mu)
        O.bodies[left_wall].material.frictionAngle = mu
        O.bodies[bottom_wall].material.frictionAngle = wallMu
        # Switch to another function to check when the collapse is finished
        check_unbalanced.command = 'check_unbalanced_after_collapse()'
        check_unbalanced.dead = True
        print('Right wall removed, column collapse started.')
        # Activate the measurement of run-out distance
        measure_run_out.dead = False

def check_unbalanced_after_collapse():
    # Check if the system has stabilized after the collapse
    if unbalancedForce() < stabilityRatio:
        print(f'Collapse finished, system is stable.')
        O.pause()

def measure_run_out_distance():
    global width, height
    # Measure the run-out distance of the collapsed column
    run_out_distance = max([b.state.pos[0] + b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)])
    # Measure the height of the collapsed column
    height = max([b.state.pos[1] + b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)])
    print(f'Run-out distance: {run_out_distance}')
    print(f'Final height: {height}')
    # Measure center of mass
    com = sum((b.state.pos * b.state.mass for b in O.bodies if isinstance(b.shape, Sphere)), Vector3(0, 0, 0))
    total_mass = sum((b.state.mass for b in O.bodies if isinstance(b.shape, Sphere)))
    com /= total_mass
    # Add to plot data
    plot.addData(t=O.time,
                 run_out=run_out_distance,
                 final_height=height,
                 com_x=com[0],
                 com_y=com[1])
    # Save VTK output
    if export_VTK:
        vtkExport.exportSpheres()
    # Optionally compute coarse-grained fields if dependencies are available
    if export_CG:
        write_particle_data()

    # finalize when we have collected the target number of points
    if measure_run_out.nDone == target_num_points:
        print('Target number of measurement points reached; finalizing output files...')
        # write simulation and parameter data in calibration-friendly format
        data_file_name = f"{description}_sim.txt"
        data_param_name = f"{description}_param.txt"
        # initialize parameter dictionary from YADE table
        param_data = {}
        for name in table.__all__:
            param_data[name] = eval('table.' + name)
        # write out simulation time series and parameters
        write_dict_to_file(plot.data, data_file_name)
        write_dict_to_file(param_data, data_param_name)
        O.pause()

def write_particle_data():
    import sys
    sys.path.append("/home/jovyan")
    from CG import UniformGrid, coarse_grain_weinhart
    from plotting import plot_scalars_2d, plot_vector_field_2d, plot_stress_2d
    from checks import check_mass_momentum_conservation
    d = 0.01
    dx = 0.02
    dy = 0.01
    nx = ny = 100
    # write particle data into numpy arrays
    ids = np.array([b.id for b in O.bodies if isinstance(b.shape, Sphere)])
    position = np.array([b.state.pos.xy() for b in O.bodies if isinstance(b.shape, Sphere)])
    mass = np.array([b.state.mass for b in O.bodies if isinstance(b.shape, Sphere)])
    radii = np.array([b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)])
    velocity = np.array([b.state.vel.xy() for b in O.bodies if isinstance(b.shape, Sphere)])
    pos_ref = np.array([b.state.displ().xy() for b in O.bodies if isinstance(b.shape, Sphere)])
    # write interaction data into numpy arrays
    contacts_i = []
    contacts_j = []
    contact_forces = []
    for i in O.interactions:
        if isinstance(O.bodies[i.id1].shape, Sphere) and isinstance(O.bodies[i.id2].shape, Sphere):
            contacts_i.append(i.id1)
            contacts_j.append(i.id2)
            contact_forces.append(i.phys.normalForce.xy() + i.phys.shearForce.xy())
    contacts_i = np.array(contacts_i)
    contacts_j = np.array(contacts_j)
    contact_forces = np.array(contact_forces)
    
    grid = UniformGrid(origin=(0,0), spacing=(dx,dy), shape=(nx,ny))
    out = coarse_grain_weinhart(
        grid,
        ids=ids, pos=position, mass=mass, vel=velocity, pos_ref=pos_ref, radii=radii,
        contacts_i=contacts_i, contacts_j=contacts_j, contact_forces=contact_forces,
        w_len=(1.5*d, 1.5*d), cutoff_c=3.0, periodic=False,
        compute_scalars=True, compute_vectors=True, compute_stress=True, stress_quad=3
    )
    # # save rho and phi as PNGs in ./figures/
    # plot_scalars_2d(grid, out["scalars"], keys=["rho","phi","occ"], save="figs/scalars")
    # plot_vector_field_2d(grid, out["vectors"]["vel"], component=None, save="figs/vectors/vel_norm")
    # plot_vector_field_2d(grid, out["vectors"]["vel"], component=0, save="figs/vectors/vel_x")
    # plot_vector_field_2d(grid, out["vectors"]["vel"], component=1, save="figs/vectors/vel_y")
    # plot_stress_2d(grid, out["tensors"], key="xy", save="figs/tensors/stress_xy")
    # plot_stress_2d(grid, out["tensors"], key="xx", save="figs/tensors/stress_xx")
    # plot_stress_2d(grid, out["tensors"], key="yy", save="figs/tensors/stress_yy")
    # plot_stress_2d(grid, out["tensors"], key="mean", save="figs/tensors/stress_mean")
    # plot_stress_2d(grid, out["tensors"], key="deviatoric", save="figs/tensors/stress_deviatoric")
    # # sanity check on conservation
    # report = check_mass_momentum_conservation(
    #     grid,
    #     pos=position, mass=mass, vel=velocity,
    #     cg_out=out,
    #     rtol_mass=1e-6, rtol_mom=1e-4
    # )
    # save the coarse-grained fields into a npy file
    np.save(f"column_collapse_{description}_{O.iter}_fields.npy", out)
    return out

# define a VTK recorder
vtkExport = VTKExporter(f'column_collapse_{description}')
export_VTK = False  # whether to export VTK files during the simulation
export_CG = False   # whether to export coarse-grained fields during the simulation

# run in batch mode
O.run()
waitIfBatch()

# create a directory to save coarse-grained fields
output = {}