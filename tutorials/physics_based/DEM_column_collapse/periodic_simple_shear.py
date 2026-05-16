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
    # wall friction
    wallMu=90,
    # confining pressure
    conf=1.0e5,
    # shear rate
    rate=1.0e-1,
    # number of particles
    num=1000,
    unknownOk=True
)

import numpy as np
from yade.params import table
from yade import pack, plot, utils
from yade.export import VTKExporter
import sys
#sys.path.append('/home/jovyan/GL/GrainLearning')
from grainlearning.tools import get_keys_and_data, write_dict_to_file

# check if run in batch mode
isBatch = runningInBatch()
if isBatch:
    description = O.tags['description']
else:
    description = 'simple_shear_DEM_test_run'

#: Domain size
width = 0.7
height = 4
depth = 0.7

#: Simulation control parameters
num = table.num  # number of soil particles
e = 0.5  # initial void ratio
damp = 0.2  # damping coefficient
stabilityRatio = 1.e-3  # threshold for quasi-static condition (decrease this for serious calculations)
output_period = 5000   # output data every n steps
target_num_points = 500   # target number of snapshots
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

#: create materials (rolling resistance and cohesion are deactivated for now)
spMat = O.materials.append(
    CohFrictMat(young=E, poisson=v, frictionAngle=ctrMu, density=rho, isCohesive=False,
                alphaKr=kr, alphaKtw=kr, momentRotationLaw=False, etaRoll=eta, etaTwist=eta, label='soil'))

wallMat = O.materials.append(
    CohFrictMat(young=E, poisson=v, frictionAngle=wallMu, density=rho, isCohesive=False,
                momentRotationLaw=False, label='walls'))


# create empty sphere packing
sp = pack.SpherePack()
if create_packing:
    # generate randomly spheres with uniform radius distribution
    sp.makeCloud((mn[0], mn[1], (mn+mx)[2]), (mx[0], mx[1], (mn+mx)[2]), rMean=.01, rRelFuzz=.1, num=1000)
    # add the sphere pack to the simulation
    sp.toSimulation(material=spMat)
else:
    sp.load(f"/home/hcheng/GrainLearning/grainLearning/tutorials/physics_based/DEM_column_collapse/initial_packing_0.175.txt")
    sp.translate(Vector3(0, 0, 0.5 * depth))  # shift to the middle of the domain
    sp.toSimulation(material=spMat)

#: Create a periodic box
min_pos, max_pos = aabbExtrema()
O.periodic = True
O.cell.hSize = Matrix3(max_pos[0] - min_pos[0], 0, 0, 0, (max_pos[1] - min_pos[1]), 0, 0, 0, depth)

# make it quasi-2D by blocking motion in z direction
for b in O.bodies:
        if isinstance(b.shape, Sphere):
             b.state.blockedDOFs = 'zXY'  # make it quasi-2D

#: Define the engines
O.engines = [
        ForceResetter(),
        InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Facet_Aabb()], allowBiggerThanPeriod=True),
        InteractionLoop(
                # handle sphere+sphere and facet+sphere collisions
                [Ig2_Sphere_Sphere_ScGeom6D(), Ig2_Facet_Sphere_ScGeom6D()],
                [Ip2_CohFrictMat_CohFrictMat_CohFrictPhys(
                    frictAngle=MatchMaker(matches=((spMat, wallMat, wallMu),)),
                    )],
                [Law2_ScGeom6D_CohFrictPhys_CohesionMoment(
                        always_use_moment_law=False,
                        useIncrementalForm=True,),
                Law2_ScGeom_FrictPhys_CundallStrack()],
        ),
        GlobalStiffnessTimeStepper(timestepSafetyCoefficient=0.2, label='time_stepper'),
        NewtonIntegrator(damping=damp, gravity=Vector3(0, 0, 0), label='newton'),
        PeriTriaxController(label='triax',
                        # whether they are strains or stresses
                        stressMask=3,
                        # confining stress
                        goal=Vector3(-table.conf, -table.conf, 0) * depth,
                        # strain rate
                        maxStrainRate=(10. * table.rate, 10. * table.rate, 0),
                        # shift particles and add two parallel walls to the top and bottom
                        doneHook="triax.dead=True; add_walls_and_shift_particles()",
                        ),
        PyRunner(command="check_pressure_on_top_wall()",
                iterPeriod=output_period,
                dead=True,
                label='check_pressure'),
        PyRunner(command="measure_stress_strain()",
                iterPeriod=output_period,
                dead=True,
                label='measure')
]


# add two parallel walls to the top and bottom of the domain, and shift particles to the middle to create a simple shear setup
def add_walls_and_shift_particles():
    global bottom_wall_particles, top_wall_particles, shift
    min_pos, max_pos = aabbExtrema()
    # reset the size of the periodic box and double the sample height to create more space for shear deformation
    O.cell.hSize = Matrix3(O.cell.hSize[0,0], 0, 0, 0, 2 * (max_pos[1] - min_pos[1]), 0, 0, 0, O.cell.hSize[2,2])
    # shift the packing to the middle of the new box
    shift = Vector3(0, 0.5 * (max_pos[1] - min_pos[1]), 0)
    for b in O.bodies:
        if isinstance(b.shape, Sphere):
            b.state.pos += shift
    # select two layers of particles close to the top and bottom boundaries
    wall_width = np.mean([b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]) * 6
    bottom_wall_particles = [b for b in O.bodies if isinstance(b.shape, Sphere) and b.state.pos[1] < (min_pos[1] + shift[1] + wall_width)]
    top_wall_particles = [b for b in O.bodies if isinstance(b.shape, Sphere) and b.state.pos[1] > (max_pos[1] + shift[1] - wall_width)]
    # set their velocities to zero and fix their degrees of freedom
    for b in bottom_wall_particles + top_wall_particles:
        b.state.vel = Vector3(0, 0, 0)
        b.state.blockedDOFs = 'xyzXYZ'
        b.shape.color = (1, 0, 0)
    # Set higher friction for the rest of the simulation
    setContactFriction(mu)
    # start moving the top wall to apply shear
    move_top_wall()
    # deactive automatic timestepper
    time_stepper.dead = True
    O.dt = 4.720963327098508e-06
    # activate the stress/strain measurement
    measure.dead = False

# # maintain constant pressure on the top wall by adjusting its velocity based on the measured pressure
# def check_pressure_on_top_wall():
#     force = O.bodies[top_wall_id].state.force
#     area = O.cell.hSize[0,0]
#     pressure = -force[1] / area
#     error = pressure - table.conf
#     if abs(error) > 0.01 * table.conf:
#         # simple proportional controller to adjust the velocity of the top wall
#         velocity_adjustment = 0.1 * error
#         O.bodies[top_wall_id].state.vel = Vector3(0, velocity_adjustment, 0)
#     else:
#         check_pressure.dead = True
#         measure_stress_strain.dead = False
#         print(f"Target pressure reached: {pressure:.2f}. Now only move the wall horizontally.")
#         min_pos, max_pos = aabbExtrema()
#         setRefSe3()
#         O.bodies[top_wall_id].state.vel = Vector3(table.rate * (max_pos[1] - min_pos[1]), 0, 0)
#     print(f"Time: {O.time:.4f}, Pressure on top wall: {pressure:.2f}, Velocity of top wall: {O.bodies[top_wall_id].state.vel}")

def move_top_wall():
    s = getStress()
    pressure = (s[0, 0] + s[1, 1] )/ 2
    print(f"Target pressure reached: {pressure:.2f}. Now only move the wall horizontally.")
    min_pos, max_pos = aabbExtrema()
    setRefSe3()
    for b in top_wall_particles:
        b.state.vel = Vector3(table.rate * (max_pos[1] - min_pos[1]), 0, 0)

# measure stress and strain during the shear phase
def measure_stress_strain():
    # Measure stresses on the periodic sides (horizontal)
    stress = getStress() * depth
    s_xx = stress[0, 0]
    s_xy = stress[0, 1]
    s_yy = stress[1, 1]
    s_yx = stress[1, 0]
    # Add to plot data
    plot.addData(t=O.time,
                 s_xx=s_xx,
                 s_xy=s_xy,
                 s_yy=s_yy,
                 s_yx=s_yx,
)
    # Save VTK output
    if export_VTK:
        vtkExport.exportSpheres()
    # Optionally compute coarse-grained fields if dependencies are available
    if export_CG:
        write_particle_data()

    # finalize when we have collected the target number of points
    if measure.nDone == target_num_points:
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
    # sys.path.append("/home/jovyan")
    from grainlearning.coarse_graining.CG import coarse_grain, UniformGrid
    from grainlearning.coarse_graining.plotting import plot_scalars_2d, plot_vector_field_2d, plot_stress_2d
    from grainlearning.coarse_graining.checks import check_mass_momentum_conservation
    d = 0.01
    # get lower left corner of the domain for reference
    min_pos, max_pos = aabbExtrema()
    nx = ny = 100
    dx = O.cell.hSize[0, 0] / nx
    dy = (max_pos[1] - min_pos[1]) / ny

    # write particle data into numpy arrays
    ids = np.array([b.id for b in O.bodies if isinstance(b.shape, Sphere)])
    position = np.array([O.cell.wrap(b.state.pos).xy() for b in O.bodies if isinstance(b.shape, Sphere)])
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
        
    grid = UniformGrid(origin=(0,shift[1]-3*d), spacing=(dx,dy), shape=(nx,ny))
    out = coarse_grain(
        grid,
        ids=ids, pos=position, mass=mass, vel=velocity, pos_ref=pos_ref, radii=radii,
        contacts_i=contacts_i, contacts_j=contacts_j, contact_forces=contact_forces,
        w_len=(1.5*d, 1.5*d), cutoff_c=3.0, periodic=[True, False],
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
    np.save(f"{description}_{O.iter}_CG_fields.npy", out)
    return out

# define a VTK recorder
vtkExport = VTKExporter(f'simple_shear_{description}')
export_VTK = False  # whether to export VTK files during the simulation
export_CG = True   # whether to export coarse-grained fields during the simulation

# run in batch mode
O.run()
waitIfBatch()