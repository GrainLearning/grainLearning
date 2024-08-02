import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import sys
from scipy.interpolate import interp1d
from scipy.linalg import cholesky
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from multiprocessing import Pool


# function list:

# function 1: partitioning of the FEM domain
# function 2: 8 node quadrilateral mesh generator
# function 3: elevate mesh to form the free surface that was pre-defined
# function 4: calculate the equivalent thickness of the pile
# function 5: loop over surface's to find intersection points
# function 6: find intersection point between the free surface and the water level
# function 7: calculate cross product between two line in 2D
# function 8: calculate additional stiffness due to anchor presence
# function 9: find closest node on pile where anchor should be present
# function 10: define element degree of freedom array
# function 11: find dirichlet boundary node's
# function 12: find Neumann boundary (due to water pressure)
# function 13: find outward normal to a boundary edge
# function 14: find integration point location
# function 15: shape function 4 node quadrilateral
# function 16: derivate shape function 4 node quadrilateral
# function 17: shape function 8 node quadrilateral
# function 18: derivate shape function 8 node quadrilateral
# function 19: local coordinates integration points
# function 20: generate random field, (Gaussian correlation fuction)
# function 21: set Youngs modulus and Poisson's ratio throughout FEM domain
# function 22: find interface integration points
# function 23: find pile integration points
# function 24: form global derivative matrix
# function 25: calculate Lamé parameters
# function 26: calculate strain-displacement matrix
# function 27: form displacement stiffness matrix
# function 28: set specific weight throughout FEM domain
# function 29: form body load vector
# function 30: form init head based on water surface wl
# function 31: random field instance for friction angle, cohesion and hydraulic conductivity
# function 32: log normal distribution
# function 33: set hydraulic conductivity throughout FEM domain
# function 34: set plasticity parameters throughout FEM domain
# function 35: form groundwater stiffness matrix
# function 36: FEM solver for groundwater flow
# function 37: calculate Darcy flow
# function 38: hydromechanical coupling porepressure to total stress
# function 39: vectorized version of nodal coordinate array
# function 40: vectorized version of array declarations displacement solver array
# function 41: vectorized version of displacement vector array
# function 42: vectorized version of strain calculation
# function 43: vectorized version of stress calculation
# function 44: vectorized version of indexing array
# function 45: shear strength reduction routine
# function 46: vectorized version stress invariant routine
# function 47: vectorized version of Mohr Coulomb Yield function & derivatives
# function 48: vectorized version of the global derivative array
# function 49: select pile elements routine
# function 50: vectorized solver for the displacement and stability calculation
# function 51: find safety factor by executing displacement solvers with several factors of safety
# function 52: display Monte Carlo results
# function 53: display single run results
# function 54: single Monte Carlo iteration (groundwater-displacement-factor of safety)
# function 55: parallel task Monte Carlo Iterations
# function 56: parallel execution Monte Carlo Iterations
# function 57: non-vectorized solver for the displacement and stability calculation

# function 1: partitioning of the FEM domain
def divide_FEM_domain(free, disc, pile, equivalent_pile_thickness, intersection_point_list, active_pile):
    # Calculate discretization in the y-direction
    ey = np.max(free[:, 1]) / disc[1]
    # Determine the Y-coordinates
    if active_pile:
        # If the pile is active, include specific ratio related to the pile position
        Y = np.unique(np.concatenate(
            (np.linspace(0, 1.0, int(ey) + 1), [(free[(pile[0] - 1), 1] - pile[1]) / free[(pile[0] - 1), 1]])))
    else:
        # If the pile is not active, use a linearly spaced array
        Y = np.linspace(0, 1.0, int(ey) + 1)
    # Initialize X as the x-coordinates in `free`
    X = np.array(free[:, 0])
    # Iteratively add x-coordinates, dividing each segment into equally spaced intervals
    for i in range(free.shape[0] - 1):
        L = free[i + 1, 0] - free[i, 0]
        ex = int(np.ceil(L / disc[0]))
        dx = L / ex
        for j in range(ex):
            X = np.append(X, free[i, 0] + j * dx)
    # Add additional point for active pile considering equivalent pile thickness
    if active_pile:
        X = np.append(X, free[(pile[0] - 1), 0] + equivalent_pile_thickness)
    # Include intersection points if they exist
    if len(intersection_point_list) != 0:
        X = np.concatenate([X, intersection_point_list[:, 0]])
    # Ensure X contains only unique values
    X = np.unique(X)
    # Return the discretized coordinates in the FEM domain
    return X, Y


# function 2: 8 node quadrilateral mesh generator
def mesh_generator(X, Y):
    # Calculate the number of elements along the x and y axes
    ex = len(X) - 1
    ey = len(Y) - 1
    # Calculate the total number of elements and nodes
    ec = ex * ey
    nx = ex + 1
    ny = ey + 1
    # Calculate the total number of nodes with 8 node quadrilateral mesh
    nc = (2 * ex + 1) * (2 * ey + 1) - ec
    # Initialize arrays for elements and nodes
    el = np.zeros((ec, 8), dtype=int)
    no = np.zeros((nc, 2))
    # Loop through each element to define node positions and element connectivity
    for i in range(ey):
        for j in range(ex):
            # Calculate the index of the current element
            el_index = i * ex + j
            # Define the node indices for the current element
            el[el_index, :] = [
                i * nx + j,
                (i + 1) * nx + j,
                (i + 1) * nx + j + 1,
                i * nx + j + 1,
                nx * ny + ex + i * (nx + ex) + j,
                nx * ny + (i + 1) * (nx + ex) + j,
                nx * ny + ex + i * (nx + ex) + j + 1,
                nx * ny + i * (nx + ex) + j
            ]
            # Define the x and y coordinates for each node in the current element
            el_nodes = el[el_index, :]
            no[el_nodes, 0] = [X[j], X[j], X[j + 1], X[j + 1], X[j], (X[j] + X[j + 1]) / 2, X[j + 1],
                               (X[j] + X[j + 1]) / 2]
            no[el_nodes, 1] = [Y[i], Y[i + 1], Y[i + 1], Y[i], (Y[i] + Y[i + 1]) / 2, Y[i + 1], (Y[i] + Y[i + 1]) / 2,
                               Y[i]]
    # Return the node positions, the number of corners, the elements, and the number of elements
    return no, nc, el, ec


# function 3: elevate mesh to form the free surface that was pre-defined
def elevate_mesh(free, X, no, el):
    # Calculate the number of nodes of the 4 node quadrilateral mesh
    nc = np.max(el[:, :4]) + 1
    nx = len(X)
    ex = nx - 1
    # Interpolate the free surface using the interp1d function
    interp_function = interp1d(free[:, 0], free[:, 1], fill_value="extrapolate")
    free_surface = interp_function(X)
    # Calculate the midpoints of the free surface segments
    free_surface = np.concatenate((free_surface, (free_surface[:-1] + free_surface[1:]) / 2))
    # Adjust the y-coordinates of nodes based on the free surface elevation (4 node quadrilateral)
    for i in range(nx):
        no[i:nc:nx, 1] *= free_surface[i]
    # Adjust left and right mid node (8 node quadrilateral)
    for i in range(nx):
        no[nc + ex + i:no.shape[0]: ex + nx, 1] *= free_surface[i]
    # Adjust up and down node (8 node quadrilateral)
    for i in range(ex):
        no[nc + i:no.shape[0]:ex + nx, 1] *= free_surface[nx + i]
    # Identify nodes that are on the free surface
    free_surface_node = np.concatenate([np.arange(nc - ex - 1, nc), np.arange(no.shape[0] - ex, no.shape[0])])
    id_sorted = np.argsort(no[free_surface_node, 0])
    free_surface_node = free_surface_node[id_sorted]
    # Return the modified node positions and indices of nodes on the free surface
    return no, free_surface_node


# function 4: calculate the equivalent thickness of the pile
def equivalent_thickness_rectangle(I):
    equivalent_pile_thickness = (12 * I) ** (1 / 3)
    return equivalent_pile_thickness


# function 5: loop over surface's to find intersection points
def find_intersection_points(free, wl):
    # Initialize a list to store intersection points
    intersection_point_list = []
    # Iterate through each segment in 'free'
    for i in range(free.shape[0] - 1):
        # Iterate through each segment in 'wl' (water line)
        for j in range(wl.shape[0] - 1):
            # Check for intersection between a segment in 'free' and a segment in 'wl'
            intersection_point = cross_check(free[i, :], free[i + 1, :], wl[j, :], wl[j + 1, :])
            # If an intersection point is found, add it to the list
            if intersection_point is not None:
                intersection_point_list.append(intersection_point)
    # Return the list of intersection points as a numpy array
    return np.array(intersection_point_list)


# function 6: find intersection point between the free surface and the water level
def cross_check(p1, p2, p3, p4):
    # Calculate the vectors for the two line segments
    v1 = p2 - p1
    v2 = p4 - p3
    # Calculate the denominator using a 2D cross product function
    denom = cross_product_2d(v1, v2)
    # If the denominator is zero, the lines are parallel or coincident, so return None
    if denom == 0:
        return None
    # Calculate a vector from one line's start point to the other's
    v3 = p3 - p1
    # Calculate the intersection parameters for both lines
    t1 = cross_product_2d(v3, v2) / denom
    t2 = cross_product_2d(v3, v1) / denom
    # Check if the intersection point lies within both line segments
    if 0 < t1 < 1 and 0 < t2 < 1:
        # Calculate the intersection point
        intersection_point = p1 + t1 * v1
        return intersection_point
    else:
        # If there is no valid intersection within both line segments, return None
        return None


# function 7: calculate cross product between two line in 2D
def cross_product_2d(v1, v2):
    crossp = v1[0] * v2[1] - v1[1] * v2[0]
    return crossp


# function 8: calculate additional stiffness due to anchor presence
def anchor_spring(free, pile, anchor, E_anchor, d_anchor, HoH_anchor, no, active_anchor):
    # Check if the anchor is active
    if active_anchor:
        # Select the appropriate anchor node based on the provided criteria
        anchor_id = select_anchor_node(free, pile, anchor, no)
        # Calculate the cross-sectional area of the anchor
        A_anchor = np.pi * (d_anchor / 2) ** 2
        # Compute the distances in x and y directions from the anchor node to the anchor point
        dx = no[anchor_id, 0] - anchor[1]
        dy = no[anchor_id, 1] - anchor[2]
        # Calculate the length of the anchor
        L_anchor = np.sqrt(dx ** 2 + dy ** 2)
        # Determine the angle of the anchor with respect to the horizontal
        angle_anchor = np.degrees(np.arcsin(dy / L_anchor))
        # Calculate the equivalent spring constant for the anchor
        equivalent_anchor_spring = E_anchor * A_anchor / (L_anchor * HoH_anchor)
        # Decompose the spring constant into x and y components
        anchor_spring_x = equivalent_anchor_spring * np.abs(np.cos(np.radians(angle_anchor)))
        anchor_spring_y = equivalent_anchor_spring * np.abs(np.sin(np.radians(angle_anchor)))
    else:
        # If the anchor is not active, set all outputs to empty arrays
        anchor_id = np.array([])
        anchor_spring_x = np.array([])
        anchor_spring_y = np.array([])

    # Return the anchor node ID and the spring constants in the x and y directions
    return anchor_id, anchor_spring_x, anchor_spring_y


# function 9: find closest node on pile with anchor point
def select_anchor_node(free, pile, anchor, no):
    distances = np.sqrt((no[:, 0] - free[(pile[0] - 1), 0]) ** 2 + (no[:, 1] - anchor[0]) ** 2)
    id = np.argmin(distances)
    return id


# function 10: define element degree of freedom array
def degree_freedom(el, ec):
    df = np.zeros((ec, 16), dtype=int)
    for i in range(8):
        df[:, i * 2] = el[:, i] * 2
        df[:, i * 2 + 1] = el[:, i] * 2 + 1
    return df


# function 11: find dirichlet boundary node's
def boundary_dirichlet(wl, X, no, nc, el, free_surface_node):
    # Find node indices at the minimum and maximum X positions and at Y=0
    x0 = np.where(no[:, 0] == np.min(X))[0]
    x1 = np.where(no[:, 0] == np.max(X))[0]
    y0 = np.where(no[:, 1] == 0.0)[0]
    # Create an array of restricted degrees of freedom (Dirichlet boundary conditions)
    rd = np.unique(np.concatenate([x0 * 2, x1 * 2, y0 * 2, y0 * 2 + 1]))
    # Create an array of active degrees of freedom
    nd = np.arange(1, 2 * nc)
    ad = nd[~np.isin(nd, rd)]
    # Calculate pressure at free surface nodes based on water line (wl) interpolation
    free_surface_pressure_node = free_surface_node[::2]
    interp_function = interp1d(wl[:, 0], wl[:, 1], fill_value="extrapolate")
    fsh = interp_function(no[free_surface_pressure_node, 0])
    # Determine fixed head nodes where wl > free and left and right boundary's
    el_max = np.max(el[:, :4]) + 1
    fh1 = free_surface_pressure_node[fsh >= no[free_surface_pressure_node, 1]]
    fh2 = np.where((no[:el_max, 0] == np.min(X)) & (no[:el_max, 1] <= fsh[0]))[0]
    fh3 = np.where((no[:el_max, 0] == np.max(X)) & (no[:el_max, 1] <= fsh[-1]))[0]
    # Determine possible seepage head nodes where wl < free and left and right boundary's
    fs1 = free_surface_pressure_node[fsh < no[free_surface_pressure_node, 1]]
    fs2 = np.where((no[:el_max, 0] == np.min(X)) & (no[:el_max, 1] > fsh[0]))[0]
    fs3 = np.where((no[:el_max, 0] == np.max(X)) & (no[:el_max, 1] > fsh[-1]))[0]
    # Concatenate node indices for hydrostatic and free surface
    fh = np.concatenate([fh1, fh2, fh3])
    fs = np.concatenate([fs1, fs2, fs3])
    # Return the arrays of active degrees of freedom, hydrostatic and free surface nodes
    return ad, fh, fs


# function 12: find Neumann boundary (due to water pressure)
def boundary_neumann(wl, no, nc, free_surface_node):
    # Interpolate the water level at the free surface nodes
    interp_function = interp1d(wl[:, 0], wl[:, 1], fill_value="extrapolate")
    free_surface_water_level = interp_function(no[free_surface_node, 0])
    # Calculate the depth of water at each free surface node
    water_depth = free_surface_water_level - no[free_surface_node, 1]
    # Compute the mean water depth between pairs of nodes
    mean_water_depth = (water_depth[::2][:-1] + water_depth[2::2]) / 2
    # Compute the length and normal vectors at each free surface segment
    dL, nxx, nyy = normal_boundary(no[free_surface_node[::2][:-1]], no[free_surface_node[2::2]])
    # Identify indices for distributing water pressure on nodes
    id1 = np.arange(0, len(water_depth) - 2, 2)
    id2 = np.arange(1, len(water_depth) - 1, 2)
    id3 = np.arange(2, len(water_depth), 2)
    # Calculate water pressure based on mean water depth
    water_pressure = 10.0 * mean_water_depth
    water_pressure[water_pressure <= 0.0] = 0.0
    # Initialize an array for nodal water pressure
    nodal_water_pressure = np.zeros(2 * nc)
    # Distribute the water pressure to the nodes of the 8 node quadrilaterals
    nodal_water_pressure[free_surface_node[id1] * 2] -= water_pressure / 6.0 * dL * nxx
    nodal_water_pressure[free_surface_node[id1] * 2 + 1] -= water_pressure / 6.0 * dL * nyy
    nodal_water_pressure[free_surface_node[id2] * 2] -= 2 * water_pressure / 3.0 * dL * nxx
    nodal_water_pressure[free_surface_node[id2] * 2 + 1] -= 2 * water_pressure / 3.0 * dL * nyy
    nodal_water_pressure[free_surface_node[id3] * 2] -= water_pressure / 6.0 * dL * nxx
    nodal_water_pressure[free_surface_node[id3] * 2 + 1] -= water_pressure / 6.0 * dL * nyy
    # Return the calculated nodal water pressure
    return nodal_water_pressure


# function 13: find outward normal to a boundary edge
def normal_boundary(n1, n2):
    # Calculate the difference in x and y coordinates between pairs of nodes
    dx = n2[:, 0] - n1[:, 0]
    dy = n2[:, 1] - n1[:, 1]
    # Compute the length of the segment between each pair of nodes
    dL = np.sqrt(dx ** 2 + dy ** 2)
    # Calculate the normal vector components for each segment
    # The normal vector is perpendicular to the segment
    nxx = -dy / dL
    nyy = dx / dL
    # Return the length of each segment and the normal vector components
    return dL, nxx, nyy


# function 14: find integration point location
def integration_point_location(no, el, ec):
    # Obtain integration points in local coordinates (xi, yi)
    xi, yi = integration_point()
    # Initialize an array to store the locations of integration points
    lip = np.zeros((4 * ec, 2))
    # Iterate over each element in the mesh
    for i in range(ec):
        # Get the coordinates of the nodes of the current element
        co = no[el[i, :], :]
        # Iterate over each integration point (4 per element in this case)
        for j in range(4):
            # Compute the shape tensor at each integration point
            N = shape_tensor8(xi[j], yi[j])
            # Calculate the global coordinates of the integration point
            lip[i * 4 + j, :] = np.dot(N, co)
    # Return the global coordinates of all integration points
    return lip


# function 15: shape function 4 node quadrilateral
def shape_tensor4(xi, yi):
    # Calculate the modified coordinates for interpolation
    xim = 1 - xi
    xip = 1 + xi
    yim = 1 - yi
    yip = 1 + yi
    # Compute the shape functions for a 4-node quadrilateral element
    N1 = 0.25 * xim * yim
    N2 = 0.25 * xim * yip
    N3 = 0.25 * xip * yip
    N4 = 0.25 * xip * yim
    # Combine the shape functions into an array
    N = np.array([N1, N2, N3, N4])
    # Return the array of shape functions
    return N


# function 16: derivate shape function 4 node quadrilateral
def local_derivative4(xi, yi):
    # Calculate the modified coordinates for differentiation
    xim = 1 - xi
    xip = 1 + xi
    yim = 1 - yi
    yip = 1 + yi
    # Compute the derivatives of the shape functions with respect to xi
    dN1_1 = -0.25 * yim
    dN1_2 = -0.25 * yip
    dN1_3 = 0.25 * yip
    dN1_4 = 0.25 * yim
    # Compute the derivatives of the shape functions with respect to yi
    dN2_1 = -0.25 * xim
    dN2_2 = 0.25 * xim
    dN2_3 = 0.25 * xip
    dN2_4 = -0.25 * xip
    # Combine the derivatives into a 2x4 array
    dN = np.array([[dN1_1, dN1_2, dN1_3, dN1_4],
                   [dN2_1, dN2_2, dN2_3, dN2_4]])
    # Return the array of shape function derivatives
    return dN


# function 17: shape function 8 node quadrilateral
def shape_tensor8(xi, yi):
    # Calculate the modified coordinates for interpolation
    xim = 1 - xi
    xip = 1 + xi
    yim = 1 - yi
    yip = 1 + yi
    # Compute the shape functions for an 8-node quadrilateral element
    N1 = 0.25 * xim * yim * (-xi - yi - 1)
    N2 = 0.25 * xim * yip * (-xi + yi - 1)
    N3 = 0.25 * xip * yip * (xi + yi - 1)
    N4 = 0.25 * xip * yim * (xi - yi - 1)
    N5 = 0.5 * xim * yim * yip
    N6 = 0.5 * xim * xip * yip
    N7 = 0.5 * xip * yim * yip
    N8 = 0.5 * xim * xip * yim
    # Combine the shape functions into an array
    N = np.array([N1, N2, N3, N4, N5, N6, N7, N8])
    # Return the array of shape functions
    return N


# function 18: derivate shape function 8 node quadrilateral
def local_derivative8(xi, yi):
    # Calculate the modified coordinates for differentiation
    xim = 1 - xi
    xip = 1 + xi
    yim = 1 - yi
    yip = 1 + yi
    # Compute the derivatives of the shape functions with respect to xi
    dN1_1 = 0.25 * yim * (2 * xi + yi)
    dN1_2 = 0.25 * yip * (2 * xi - yi)
    dN1_3 = 0.25 * yip * (2 * xi + yi)
    dN1_4 = 0.25 * yim * (2 * xi - yi)
    dN1_5 = -0.5 * yim * yip
    dN1_6 = -xi * yip
    dN1_7 = 0.5 * yim * yip
    dN1_8 = -xi * yim
    # Compute the derivatives of the shape functions with respect to yi
    dN2_1 = 0.25 * xim * (2 * yi + xi)
    dN2_2 = 0.25 * xim * (2 * yi - xi)
    dN2_3 = 0.25 * xip * (xi + 2 * yi)
    dN2_4 = 0.25 * xip * (2 * yi - xi)
    dN2_5 = -xim * yi
    dN2_6 = 0.5 * xim * xip
    dN2_7 = -xip * yi
    dN2_8 = -0.5 * xim * xip
    # Return all the derivatives
    return dN1_1, dN1_2, dN1_3, dN1_4, dN1_5, dN1_6, dN1_7, dN1_8, dN2_1, dN2_2, dN2_3, dN2_4, dN2_5, dN2_6, dN2_7, dN2_8


# function 19: local coordinates integration points
def integration_point():
    factor = 1 / np.sqrt(3)
    xi = factor * np.array([-1, 1, -1, 1])
    yi = factor * np.array([1, 1, -1, -1])
    return xi, yi


# function 20: generate random field, (Gaussian correlation fuction)
def random_field(x, y, clx, cly):
    # This function calculates the covariance based on the distance between points
    cfun = lambda x1, y1, x2, y2: np.exp(-np.sqrt(((x1 - x2) ** 2 / clx ** 2) + ((y1 - y2) ** 2 / cly ** 2)))
    # Create a meshgrid of x and y coordinates for calculating the covariance matrix
    X, Y = np.meshgrid(x, y)
    # The covariance is calculated for every pair of points in the meshgrid
    covariance_matrix = cfun(X, Y, X.T, Y.T)
    # Perform the Cholesky decomposition of the covariance matrix
    L = cholesky(covariance_matrix, lower=True)
    # Return the Cholesky factor (lower triangular matrix)
    return L


# function 21: set Youngs modulus and Poisson's ratio throughout FEM domain
def elasticity_parameters_integration_point(free, pile, inter, E_soil, E_pile, E_inter, nu_soil, nu_pile, nu_inter,
                                            equivalent_pile_thickness, ec, lip, active_pile, active_inter):
    # Initialize the elasticity modulus (E) and Poisson's ratio (nu) arrays for each integration point
    E = np.full(4 * ec, E_soil)
    nu = np.full(4 * ec, nu_soil)
    # If interfaces are active, update the elasticity parameters at relevant integration points
    if active_inter:
        # Select integration points associated with the interface
        id_inter = select_inter_integration_points(free, inter, lip)
        # Update the elasticity modulus and Poisson's ratio at these points to interface properties
        E[id_inter] = E_inter
        nu[id_inter] = nu_inter
    # If piles are active, update the elasticity parameters at relevant integration points
    if active_pile:
        # Select integration points associated with the pile
        id_pile = select_pile_integration_points(free, pile, equivalent_pile_thickness, lip)
        # Update the elasticity modulus and Poisson's ratio at these points to pile properties
        E[id_pile] = E_pile
        nu[id_pile] = nu_pile
    # Return the arrays containing the elasticity modulus and Poisson's ratio for each integration point
    return E, nu


# function 22: find interface integration points
def select_inter_integration_points(free, inter, lip):
    # Define a condition to select integration points that are within the interface region
    condition = (
        # Check if the x-coordinate of the integration point is between the x-coordinates of the interface nodes
            (lip[:, 0] >= free[(inter[0] - 1), 0]) &
            (lip[:, 0] <= free[(inter[1] - 1), 0]) &
            # Check if the y-coordinate of the integration point is within a certain range defined by the interface
            (lip[:, 1] >= (free[(inter[0] - 1), 1] + free[(inter[1] - 1), 1]) / 2 - inter[2])
    )
    # Find indices of integration points that satisfy the condition
    id = np.where(condition)[0]
    # Return the indices of the integration points within the interface region
    return id


# function 23: find pile integration points
def select_pile_integration_points(free, pile, equivalent_pile_thickness, lip):
    # Define a condition to select integration points that are within the pile region
    condition = (
        # Check if the x-coordinate of the integration point is within the x-range of the pile
            (lip[:, 0] >= free[(pile[0] - 1), 0]) &
            (lip[:, 0] <= free[(pile[0] - 1), 0] + equivalent_pile_thickness) &
            # Check if the y-coordinate of the integration point is below a certain depth defined by the pile
            (lip[:, 1] >= free[(pile[0] - 1), 1] - pile[1])
    )
    # Find indices of integration points that satisfy the condition
    id = np.where(condition)[0]
    # Return the indices of the integration points within the pile region
    return id


# function 24: form global derivative matrix
def global_derivative(dN, co):
    # Calculate the Jacobian matrix from the local derivatives and nodal coordinates
    J = np.dot(dN, co)
    # Compute the determinant of the Jacobian matrix
    # This determinant is used for calculating the area (or volume in 3D) element in the global coordinate system
    dA = np.linalg.det(J)
    # Compute the global derivative of the shape functions
    # This is done by solving the linear system J * D = dN, where D are the global derivatives
    D = np.linalg.solve(J, dN)
    # Return the global derivatives of the shape functions and the determinant of the Jacobian
    return D, dA


# function 25: calculate Lamé parameters
def elasticity_parameters(E, nu):
    # Lamé parameters 1
    Lambda = E * nu / ((1 - 2 * nu) * (1 + nu))
    # Lamé parameters 2 (shear modulus)
    Mu = E / (2 * (1 + nu))
    return Lambda, Mu


# function 26: calculate strain-displacement matrix
def strain_displacement_tensor(D):
    # Initialize a zero matrix for the strain-displacement relationship
    b = np.zeros((3, 16))
    # Fill the matrix to relate displacements to strains
    # For 2D problems, the strain-displacement matrix has 3 rows (for εx, εy, and γxy)
    # and twice the number of nodes in columns (for x and y displacements at each node)
    # The first row corresponds to strain in the x-direction (εx)
    b[0, 0:16:2] = D[0, :]
    # The second row corresponds to strain in the y-direction (εy)
    b[1, 1:16:2] = D[1, :]
    # The third row corresponds to shear strain (γxy)
    b[2, 1:16:2] = D[0, :]
    b[2, 0:16:2] = D[1, :]
    # Return the strain-displacement matrix
    return b


# function 27: form displacement stiffness matrix
def displacement_stiffness(no, nc, el, ec, df, ad, E, nu, anchorId, anchorSpringX, anchorSpringY):
    # Calculate the Lame parameters (Lambda and Mu) from elasticity modulus and Poisson's ratio
    Lambda, Mu = elasticity_parameters(E, nu)
    # Obtain integration points in local coordinates
    xi, yi = integration_point()
    # Initialize arrays for sparse matrix construction
    a1 = np.zeros(16 * 16 * ec)
    a2 = np.zeros_like(a1)
    a3 = np.zeros_like(a1)
    # Iterate over each element in the mesh
    for i in range(ec):
        # Get the coordinates of the nodes of the current element
        co = no[el[i, :], :]
        # Initialize the element stiffness matrix
        km = np.zeros((16, 16))
        # Iterate over each integration point
        for j in range(4):
            # integration point index
            id = i * 4 + j
            # Get the local derivatives for the current integration point
            dN1_1, dN1_2, dN1_3, dN1_4, dN1_5, dN1_6, dN1_7, dN1_8, dN2_1, dN2_2, dN2_3, dN2_4, dN2_5, dN2_6, dN2_7, dN2_8 = local_derivative8(
                xi[j], yi[j])
            dN = np.array([
                [dN1_1, dN1_2, dN1_3, dN1_4, dN1_5, dN1_6, dN1_7, dN1_8],
                [dN2_1, dN2_2, dN2_3, dN2_4, dN2_5, dN2_6, dN2_7, dN2_8]
            ])
            # Calculate the global derivatives and area element
            D, dA = global_derivative(dN, co)
            # Compute the strain-displacement matrix
            b = strain_displacement_tensor(D)
            # Create the elasticity matrix for the current integration point
            Ce = np.array([
                [Lambda[id] + 2 * Mu[id], Lambda[id], 0.0],
                [Lambda[id], Lambda[id] + 2 * Mu[id], 0.0],
                [0.0, 0.0, Mu[id]]
            ])
            # Accumulate the element stiffness matrix
            km += np.dot(np.dot(b.T, Ce), b) * dA
        # Store the element stiffness matrix in a linear array for sparse matrix construction
        idx = slice(i * 256, (i + 1) * 256)
        a1[idx] = np.repeat(df[i, :], 16)
        a2[idx] = np.tile(df[i, :], 16)
        a3[idx] = km.ravel()
    # Add anchor stiffness to the global matrix
    a1 = np.append(a1, anchorId * 2)
    a1 = np.append(a1, anchorId * 2 + 1)
    a2 = np.append(a2, anchorId * 2)
    a2 = np.append(a2, anchorId * 2 + 1)
    a3 = np.append(a3, anchorSpringX)
    a3 = np.append(a3, anchorSpringY)
    # Construct the global stiffness matrix as a sparse matrix
    Km = csr_matrix((a3, (a1, a2)), shape=(2 * nc, 2 * nc))
    # Convert to a Compressed Sparse Column matrix and use Sparse LU decomposition (on active degrees of freedom ad)
    Km_csc = csc_matrix(Km)
    Km_d = splu(Km_csc[ad, :][:, ad])
    # Return the decomposed stiffness matrix
    return Km_d


# function 28: set specific weight throughout FEM domain
def specificWeightIntegrationPoint(free, pile, inter, gamma_soil, gamma_pile, gamma_inter, equivalentPileThickness, ec,
                                   lip, activePile, activeInter):
    # Initialize the specific weight array for each integration point
    gamma = np.full((4 * ec, 1), gamma_soil)
    # If interfaces are active, update the specific weight at relevant integration points
    if activeInter:
        # Select integration points associated with the interface
        id = select_inter_integration_points(free, inter, lip)
        # Update the specific weight at these points to interface specific weight
        gamma[id] = gamma_inter
    # If piles are active, update the specific weight at relevant integration points
    if activePile:
        # Select integration points associated with the pile
        id = select_pile_integration_points(free, pile, equivalentPileThickness, lip)
        # Update the specific weight at these points to pile specific weight
        gamma[id] = gamma_pile
    # Return the array containing the specific weight for each integration point
    return gamma


# function 29: form body load vector
def body_weight(no, nc, el, ec, gamma):
    # Obtain integration points in local coordinates
    xi, yi = integration_point()
    # Initialize an array for nodal self-weight
    nodal_self_weight = np.zeros(2 * nc)
    # Iterate over each element in the mesh
    for i in range(ec):
        # Get the coordinates of the nodes of the current element
        co = no[el[i, :], :]
        # Iterate over each integration point
        for j in range(4):
            # Calculate the index for the specific weight at the integration point
            id = (i * 4) + j
            # Calculate the shape tensor for the integration point
            N = shape_tensor8(xi[j], yi[j])
            # Get the local derivatives for the current integration point
            dN1_1, dN1_2, dN1_3, dN1_4, dN1_5, dN1_6, dN1_7, dN1_8, dN2_1, dN2_2, dN2_3, dN2_4, dN2_5, dN2_6, dN2_7, dN2_8 = local_derivative8(
                xi[j], yi[j])
            dN = np.array([
                [dN1_1, dN1_2, dN1_3, dN1_4, dN1_5, dN1_6, dN1_7, dN1_8],
                [dN2_1, dN2_2, dN2_3, dN2_4, dN2_5, dN2_6, dN2_7, dN2_8]
            ])
            # Calculate the global derivative and area element
            D, dA = global_derivative(dN, co)
            # Distribute the self-weight to the nodal points
            # The y-component of the self-weight is considered (hence multiplying by 2 + 1)
            nodal_self_weight[el[i, :] * 2 + 1] -= N.T * gamma[id] * dA
    # Return the array of nodal self-weight
    return nodal_self_weight


# function 30: form init head based on water surface wl
def initial_head(wl, X, no, el, free_surface_node):
    # Interpolate the water level (head) at the free surface nodes using the provided water level data
    fsh = np.interp(no[free_surface_node[0::2], 0], wl[:, 0], wl[:, 1])
    # Get the number of unique x-coordinates in the mesh
    nx = len(X)
    # Initialize an array to store the initial head at each node
    H = np.zeros(np.max(el[:, 0:4]) + 1)
    # Distribute the interpolated head values to the corresponding nodes in the mesh
    for i in range(nx):
        H[i:np.max(el[:, 0:4]) + 1:nx] = fsh[i]
    # Return the array of initial head values
    return H


# function 31: random field instance for friction angle, cohesion and hydraulic conductivity
def random_field_instance(ks_soil, phi_soil, coh_soil, ec, L):
    # Generate a random field instance using the Cholesky factor (L)
    z = np.dot(L, np.random.randn(4 * ec))
    # Calculate the hydraulic conductivity (ks) using a log-normal distribution
    # The mean and variance of the log-normal distribution are based on ks_soil
    ks = log_norm_distribution(ks_soil, z)
    # Calculate the friction angle (phi) using a transformed distribution
    # The transformation ensures that phi varies within the range defined by phi_soil
    phi = phi_soil[0] + (phi_soil[1] - phi_soil[0]) * ((1 + np.tanh(z)) / 2)
    # Calculate the cohesion (coh) using a log-normal distribution
    # The mean and variance of the log-normal distribution are based on coh_soil
    coh = log_norm_distribution(coh_soil, z)
    # Return the generated values for hydraulic conductivity, friction angle, and cohesion
    return ks, phi, coh


def uniform_field_instance(ks_soil, phi_soil, coh_soil, ec):
    z = np.ones(4 * ec)
    ks = ks_soil[0] * z
    phi = phi_soil[0] * z
    coh = coh_soil[0] * z
    # Return the generated values for hydraulic conductivity, friction angle, and cohesion
    return ks, phi, coh


# function 32: log normal distribution
def log_norm_distribution(var, z):
    log_norm_sig = np.sqrt(np.log(1 + (var[1] ** 2) / (var[0] ** 2)))
    log_norm_mu = np.log(var[0]) - 0.5 * (log_norm_sig ** 2)
    var_field = np.exp(log_norm_mu + log_norm_sig * z)
    return var_field


# function 33: set hydraulic conductivity throughout FEM domain
def hydraulic_conductivity_integration_point(free, pile, ks, equivalent_pile_thickness, lip, active_pile):
    # Check if the pile is active
    if active_pile:
        # Select integration points for the pile based on the free field, pile properties,
        id = select_pile_integration_points(free, pile, equivalent_pile_thickness, lip)
        # Set hydraulic conductivity to zero at the selected integration points.
        ks[id] = 0.0
    # Return the modified hydraulic conductivity array
    return ks


# function 34: set plasticity parameters throughout FEM domain
def plasticity_parameters_integration_point(free, pile, inter, phi, phi_pile, phi_inter, dilation_factor, psi_pile,
                                            psi_inter, coh, coh_pile, coh_inter, equivalent_pile_thickness, lip,
                                            active_pile, active_inter):
    # Calculate the initial dilation angle (psi) as a product of the dilation factor and the friction angle (phi)
    psi = dilation_factor * phi
    # Check if the integration points for the interface (inter) are active
    if active_inter:
        # Select integration points for the interface based on the free field, interface properties, and lip condition
        id = select_inter_integration_points(free, inter, lip)
        # Update the friction angle (phi), dilation angle (psi), and cohesion (coh) at the interface integration points
        phi[id] = phi_inter
        psi[id] = psi_inter
        coh[id] = coh_inter
    # Check if the integration points for the pile are active
    if active_pile:
        # Select integration points for the pile based on the free field, pile properties, equivalent pile thickness, and lip condition
        id = select_pile_integration_points(free, pile, equivalent_pile_thickness, lip)
        # Update the friction angle (phi), dilation angle (psi), and cohesion (coh) at the pile integration points
        phi[id] = phi_pile
        psi[id] = psi_pile
        coh[id] = coh_pile
    # Return the updated values of friction angle, dilation angle, and cohesion
    return phi, psi, coh


# function 35: form groundwater stiffness matrix
def conductivity_stiffness(no, el, ec, ks):
    # Obtain integration points in local coordinates
    xi, yi = integration_point()
    # Initialize arrays for sparse matrix construction
    a1 = np.zeros(4 * 4 * ec)
    a2 = np.zeros_like(a1)
    a3 = np.zeros_like(a1)
    # Iterate over each element in the mesh
    for i in range(ec):
        # Get the coordinates of the nodes of the current element (4-node elements)
        co = no[el[i, 0:4], :]
        # Initialize the element conductivity matrix
        kc = np.zeros((4, 4))
        # Iterate over each integration point
        for j in range(4):
            # integration point index
            id = i * 4 + j
            # Calculate the local derivative for the current integration point
            dN = local_derivative4(xi[j], yi[j])
            # Calculate the global derivative and area element
            D, dA = global_derivative(dN, co)
            # Create the conductivity matrix for the current integration point
            ks_matrix = np.array([[ks[id], 0.0], [0.0, ks[id]]])
            # Accumulate the element conductivity matrix
            kc += D.T @ ks_matrix @ D * dA
        # Store indices for the sparse matrix construction
        indices = np.repeat(el[i, 0:4], 4)
        a1[i * 16: (i + 1) * 16] = indices
        a2[i * 16: (i + 1) * 16] = np.tile(el[i, 0:4], 4)
        a3[i * 16: (i + 1) * 16] = kc.T.flatten()
    # Get the total number of nodes
    n = np.max(el[:, 0:4]) + 1
    # Construct the global conductivity matrix as a sparse matrix
    Kc = coo_matrix((a3, (a1, a2)), shape=(n, n))
    # Return the global conductivity matrix
    return Kc


# function 36: FEM solver for groundwater flow
def hydraulic_solver(H, Kc, no, el, fh, fs, lim, tol):
    # Check if there are non-zero initial heads, zero init head's mean's no porepressure's
    if np.sum(H) > 0:
        # Define the set of all nodes
        nh = np.arange(1, np.max(el[:, 0:4]) + 1)
        # Initialize an array to store active free surface nodes
        afs = np.array([], dtype=int)
        # Identify active head nodes excluding fixed head and active free surface nodes
        ah = np.setdiff1d(nh, np.union1d(fh, afs))
        # Convert the conductivity matrix to Compressed Sparse Row format for efficient calculations
        Kc_csr = Kc.tocsr()
        # Iterative solver loop
        for k in range(lim):
            # Calculate internal flows
            Qint = Kc_csr.dot(H)
            # Copy the current head values for convergence checking
            H0 = H.copy()
            # Update the head values for active nodes
            H[ah] -= spsolve(Kc_csr[np.ix_(ah, ah)], Qint[ah])
            # Calculate the error for convergence check
            er = np.max(np.abs(H - H0)) / np.max(np.abs(H0))
            if er < tol:
                break
            elif k == lim - 1:
                raise ValueError('no conv (hydraulic phase)')
            # Check for seepage nodes that becoming active (total head > elevation)
            Hs, fsId = np.max(H[fs] - no[fs, 1]), np.argmax(H[fs] - no[fs, 1])
            if Hs > 0.0:
                afs = np.append(afs, fs[fsId])
            # Update the head for active seepage node
            H[afs] = no[afs, 1]
            # Update the list of active head nodes
            ah = np.setdiff1d(nh, np.union1d(fh, afs))
    # Return the updated head distribution
    return H


# function 37: calculate Darcy flow
def flow_darcy(H, no, el, ec, ks):
    # Obtain integration points in local coordinates
    xi, yi = integration_point()
    # Initialize arrays for Darcy velocity and flow rate
    vee = np.zeros((2, 4 * ec))
    flow_rate = np.zeros(4 * ec)
    # Iterate over each element in the mesh
    for i in range(ec):
        # Get the coordinates of the nodes of the current element (4-node elements)
        co = no[el[i, 0:4], :]
        # Iterate over each integration point
        for j in range(4):
            # Calculate the index for the hydraulic conductivity at the integration point
            id = (i * 4) + j
            # Calculate the shape tensor for the integration point
            N = shape_tensor4(xi[j], yi[j])
            # Calculate the local derivative for the current integration point
            dN = local_derivative4(xi[j], yi[j])
            # Calculate the global derivative and area element
            D, dA = global_derivative(dN, co)
            # Compute the pore water pressure at the integration point
            pw = N @ (H[el[i, 0:4]] - co[:, 1]) * 10.0
            # Calculate the Darcy velocity if the pore water pressure is positive
            if pw > 0.0:
                vee[:, id] = -np.array([[ks[id], 0.0], [0.0, ks[id]]]) @ D @ H[el[i, 0:4]]
            # Accumulate the flow rate at the nodal points
            flow_rate[el[i, 0:4]] -= D.T @ vee[:, id] * dA
    # Return the Darcy velocity and the norm of the flow rate
    return vee, flow_rate


# function 38: hydromechanical coupling porepressure to total stress
def hydro_mechanical_coupling(H, no, nc, el, ec, df):
    # Obtain integration points in local coordinates
    xi, yi = integration_point()
    # Calculate pore water pressure from the hydraulic head (H)
    P = (H - no[:np.max(el[:, :4]) + 1, 1]) * 10.0  #
    P[P < 0.0] = 0.0  # Negative pressures are not physically meaningful in this context
    # Initialize an array for nodal pore pressure
    nodal_pore_pressure = np.zeros(2 * nc)
    # Iterate over each element in the mesh
    for i in range(ec):
        # Get the coordinates of the nodes of the current element
        co = no[el[i, :], :]
        # Iterate over each integration point
        for j in range(4):
            # Calculate the shape tensor for the integration point
            N = shape_tensor4(xi[j], yi[j])
            # Get the local derivatives for the current integration point
            dN1_1, dN1_2, dN1_3, dN1_4, dN1_5, dN1_6, dN1_7, dN1_8, dN2_1, dN2_2, dN2_3, dN2_4, dN2_5, dN2_6, dN2_7, dN2_8 = local_derivative8(
                xi[j], yi[j])
            dN = np.array([
                [dN1_1, dN1_2, dN1_3, dN1_4, dN1_5, dN1_6, dN1_7, dN1_8],
                [dN2_1, dN2_2, dN2_3, dN2_4, dN2_5, dN2_6, dN2_7, dN2_8]
            ])
            # Calculate the global derivative and area element
            D, dA = global_derivative(dN, co)
            # Compute the strain-displacement matrix
            b = strain_displacement_tensor(D)
            # Distribute the pore pressure to the nodal points
            nodal_pore_pressure[df[i, :]] += b.T @ np.array([1, 1, 0]) * (N @ P[el[i, :4]]) * dA
    # Return the array of nodal pore pressure
    return nodal_pore_pressure


# function 39: vectorized version of nodal coordinate array
def coordinate_array_vectorization(no, el):
    # Extract x and y coordinates for each element node
    c1_1 = no[el[:, 0], 0]
    c1_2 = no[el[:, 0], 1]
    c2_1 = no[el[:, 1], 0]
    c2_2 = no[el[:, 1], 1]
    c3_1 = no[el[:, 2], 0]
    c3_2 = no[el[:, 2], 1]
    c4_1 = no[el[:, 3], 0]
    c4_2 = no[el[:, 3], 1]
    c5_1 = no[el[:, 4], 0]
    c5_2 = no[el[:, 4], 1]
    c6_1 = no[el[:, 5], 0]
    c6_2 = no[el[:, 5], 1]
    c7_1 = no[el[:, 6], 0]
    c7_2 = no[el[:, 6], 1]
    c8_1 = no[el[:, 7], 0]
    c8_2 = no[el[:, 7], 1]
    # Return the extracted x and y coordinates as separate variables
    return c1_1, c1_2, c2_1, c2_2, c3_1, c3_2, c4_1, c4_2, c5_1, c5_2, c6_1, c6_2, c7_1, c7_2, c8_1, c8_2


# function 40: vectorized version of array declarations displacement solver array
def array_declaration_vectorization(nc, ec):
    # Create arrays with zeros and appropriate dimensions
    U = np.zeros(2 * nc)
    evpt1 = np.zeros(4 * ec)
    evpt2 = np.zeros(4 * ec)
    evpt3 = np.zeros(4 * ec)
    evpt4 = np.zeros(4 * ec)
    sigma = np.zeros(4 * ec)
    f1 = np.zeros(ec)
    f2 = np.zeros(ec)
    f3 = np.zeros(ec)
    f4 = np.zeros(ec)
    f5 = np.zeros(ec)
    f6 = np.zeros(ec)
    f7 = np.zeros(ec)
    f8 = np.zeros(ec)
    f9 = np.zeros(ec)
    f10 = np.zeros(ec)
    f11 = np.zeros(ec)
    f12 = np.zeros(ec)
    f13 = np.zeros(ec)
    f14 = np.zeros(ec)
    f15 = np.zeros(ec)
    f16 = np.zeros(ec)
    # Return all the created arrays
    return U, evpt1, evpt2, evpt3, evpt4, sigma, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16


# function 41: vectorized version of displacement vector array
def displacement_vectorization(U, df):
    # Extract displacement values from the U array based on the indices specified in the df array
    u1 = U[df[:, 0]]
    u2 = U[df[:, 1]]
    u3 = U[df[:, 2]]
    u4 = U[df[:, 3]]
    u5 = U[df[:, 4]]
    u6 = U[df[:, 5]]
    u7 = U[df[:, 6]]
    u8 = U[df[:, 7]]
    u9 = U[df[:, 8]]
    u10 = U[df[:, 9]]
    u11 = U[df[:, 10]]
    u12 = U[df[:, 11]]
    u13 = U[df[:, 12]]
    u14 = U[df[:, 13]]
    u15 = U[df[:, 14]]
    u16 = U[df[:, 15]]
    # Return the extracted displacement values as separate variables
    return u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16


# function 42: vectorized version of strain calculation
def strain_displacement_vectorized(u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, evpt1, evpt2,
                                   evpt3, evpt4, D1_1, D1_2, D1_3, D1_4, D1_5, D1_6, D1_7, D1_8, D2_1, D2_2, D2_3, D2_4,
                                   D2_5, D2_6, D2_7, D2_8):
    # Calculate the strain components based on the displacement field and global derivatives
    # e1, e2, and e3 correspond to the normal strains in x, y, and shear strain respectively
    e1 = D1_1 * u1 + D1_2 * u3 + D1_3 * u5 + D1_4 * u7 + D1_5 * u9 + D1_6 * u11 + D1_7 * u13 + D1_8 * u15
    e2 = D2_1 * u2 + D2_2 * u4 + D2_3 * u6 + D2_4 * u8 + D2_5 * u10 + D2_6 * u12 + D2_7 * u14 + D2_8 * u16
    e3 = D1_1 * u2 + D1_2 * u4 + D1_3 * u6 + D1_4 * u8 + D2_1 * u1 + D1_5 * u10 + D2_2 * u3 + D1_6 * u12 + D2_3 * u5 + D1_7 * u14 + D2_4 * u7 + D1_8 * u16 + D2_5 * u9 + D2_6 * u11 + D2_7 * u13 + D2_8 * u15
    # Adjust the strain components by the total viscoplastic plastic strain components (evpt)
    e1 -= evpt1
    e2 -= evpt2
    e3 -= evpt3
    e4 = -evpt4  # plane strain conditions
    # Return the calculated strain components
    return e1, e2, e3, e4


# function 43: vectorized version of stress calculation
def stress_strain_vectorized(Lambda, Mu, e1, e2, e3, e4):
    # Calculate the stress components based on the strain components and material properties (Lambda and Mu)
    # s1, s2, s3, and s4 correspond to the stress components in the x, y, shear, and z directions respectively
    s1 = (Lambda + 2 * Mu) * e1 + Lambda * (e2 + e4)
    s2 = (Lambda + 2 * Mu) * e2 + Lambda * (e1 + e4)
    s3 = Mu * e3
    s4 = (Lambda + 2 * Mu) * e4 + Lambda * (e1 + e2)
    # Return the calculated stress components
    return s1, s2, s3, s4


# function 44: vectorized version of indexing array
def vector_degree_freedom(df):
    vId = np.concatenate([df[:, 0], df[:, 1], df[:, 2], df[:, 3],
                          df[:, 4], df[:, 5], df[:, 6], df[:, 7],
                          df[:, 8], df[:, 9], df[:, 10], df[:, 11],
                          df[:, 12], df[:, 13], df[:, 14], df[:, 15]])
    return vId


# function 45: shear strength reduction routine
def shear_strength_reduction(FS, phi, psi, coh, E, nu):
    # reduction friction angle
    phir = np.arctan(np.tan(np.radians(phi)) / FS)
    # reduction dilation angle
    psir = np.arctan(np.tan(np.radians(psi)) / FS)
    # recalculate Mohr-Coulomb shear strength parameters
    snph = np.sin(phir)
    csph = np.cos(phir)
    snps = np.sin(psir)
    # reduction cohesion
    cohr = coh / FS
    # calculate critical time step
    dt = (4 * (1 + nu) * (1 - 2 * nu)) / (E * (1 - 2 * nu + snph ** 2))
    # return critical time step and shear strength parameters
    return snph, csph, snps, cohr, dt


# function 46: vectorized version stress invariant routine
def stress_invariants(s1, s2, s3, s4):
    # Calculate the first stress invariant, p (mean stress)
    p = (s1 + s2 + s4) / 3
    # Calculate deviatoric stresses
    d1 = (2 * s1 - s2 - s4) / 3
    d2 = (2 * s2 - s1 - s4) / 3
    d3 = (2 * s4 - s1 - s2) / 3
    # Calculate the second stress invariant, J2 (magnitude of deviatoric stress tensor)
    J2 = 1 / 2 * (d1 ** 2 + d2 ** 2 + 2 * s3 ** 2 + d3 ** 2)
    # Initialize the Lode angle parameters
    id = J2 > 0.0  # Only calculate where J2 is positive
    q = np.zeros_like(s1)
    q[id] = np.sqrt(3 * J2[id])  # q is the equivalent deviatoric stress
    tp = np.zeros_like(s1)  # tp is an intermediate parameter for Lode angle calculation
    tp[id] = -13.5 * (d1[id] * d2[id] * d3[id] - d3[id] * s3[id] ** 2) / q[id] ** 3
    tp[tp >= 1] = 1  # compression limit lode angle
    tp[tp <= -1] = -1  # extension limit lode angle
    # Calculate the Lode angle, t
    t = np.arcsin(tp) / 3
    # Calculate partial derivatives for the consistency parameters (dp1, dp2, dp3, dp4)
    dp1 = 1.0
    dp2 = 1.0
    dp3 = 0.0
    dp4 = 1.0
    # Calculate partial derivatives of J2 with respect to the stress components
    dJ2_1 = d1
    dJ2_2 = d2
    dJ2_3 = 2 * s3
    dJ2_4 = d3
    # Calculate partial derivatives of J3 with respect to the stress components
    dJ3_1 = d1 ** 2 + s3 ** 2 - 2 / 3 * J2
    dJ3_2 = s3 ** 2 + d2 ** 2 - 2 / 3 * J2
    dJ3_3 = 2 * (d1 * s3 + s3 * d2)
    dJ3_4 = d3 ** 2 - 2 / 3 * J2
    # Return the stress invariants and their derivatives
    return p, q, t, dp1, dp2, dp3, dp4, dJ2_1, dJ2_2, dJ2_3, dJ2_4, dJ3_1, dJ3_2, dJ3_3, dJ3_4


# function 47: vectorized version of Mohr Coulomb Yield function & derivatives
def yield_mohr_coulomb(p, q, t, snph, csph, snps, coh):
    # Calculate trigonometric functions and constants used in the Mohr-Coulomb material description
    snth = np.sin(t)
    csth = np.cos(t)
    cs3th = np.cos(3.0 * t)
    tn3th = np.tan(3.0 * t)
    tnth = snth / csth
    sq3 = np.sqrt(3.0)
    # Calculate the Mohr-Coulomb yield function
    f = p * snph - coh * csph + q * (csth / sq3 - (snph * snth) / 3.0)
    f[f < 0.0] = 0.0
    # Calculate the derivatives of the yield function with respect to p,q and t
    dq1 = snps
    dq2 = (csth * sq3 * (tn3th * tnth + (snps * sq3 * (tn3th - tnth)) / 3.0 + 1.0)) / (2.0 * q)
    dq3 = (1.0 / q ** 2 * (3.0 * csth * snps / 2.0 + snth * 3.0 * sq3 / 2.0)) / cs3th
    # Adjust dq2 and dq3 close to pure extension and compression to avoid singularities
    mask1 = snth > 0.49
    mask2 = snth < -0.49
    dq2[mask1] = (sq3 / 2.0 - snps[mask1] / (2.0 * sq3)) * sq3 / (2.0 * q[mask1])
    dq2[mask2] = (sq3 / 2.0 + snps[mask2] / (2.0 * sq3)) * sq3 / (2.0 * q[mask2])
    dq3[mask1] = 0.0
    dq3[mask2] = 0.0
    # Calculate the shear strength, based on the Mohr-Coulomb criterion
    sigma = (-p * snph + coh * csph) / (csth / sq3 - (snph * snth) / 3.0)
    # Return the yield function, its derivatives, and the equivalent stress
    return f, dq1, dq2, dq3, sigma


# function 48: vectorized version of the global derivative array
def global_derivative_vectorized(dN1_1, dN1_2, dN1_3, dN1_4, dN1_5, dN1_6, dN1_7, dN1_8, dN2_1, dN2_2, dN2_3, dN2_4,
                                 dN2_5, dN2_6, dN2_7, dN2_8, c1_1, c1_2, c2_1, c2_2, c3_1, c3_2, c4_1, c4_2, c5_1, c5_2,
                                 c6_1, c6_2, c7_1, c7_2, c8_1, c8_2):
    # Calculate the Jacobian matrix components
    J1_1 = c1_1 * dN1_1 + c2_1 * dN1_2 + c3_1 * dN1_3 + c4_1 * dN1_4 + c5_1 * dN1_5 + c6_1 * dN1_6 + c7_1 * dN1_7 + c8_1 * dN1_8
    J1_2 = c1_2 * dN1_1 + c2_2 * dN1_2 + c3_2 * dN1_3 + c4_2 * dN1_4 + c5_2 * dN1_5 + c6_2 * dN1_6 + c7_2 * dN1_7 + c8_2 * dN1_8
    J2_1 = c1_1 * dN2_1 + c2_1 * dN2_2 + c3_1 * dN2_3 + c4_1 * dN2_4 + c5_1 * dN2_5 + c6_1 * dN2_6 + c7_1 * dN2_7 + c8_1 * dN2_8
    J2_2 = c1_2 * dN2_1 + c2_2 * dN2_2 + c3_2 * dN2_3 + c4_2 * dN2_4 + c5_2 * dN2_5 + c6_2 * dN2_6 + c7_2 * dN2_7 + c8_2 * dN2_8
    # Compute the determinant of the Jacobian matrix
    dA = J1_1 * J2_2 - J1_2 * J2_1
    # Compute the inverse Jacobian matrix components
    iJ1_1 = J2_2 / dA
    iJ1_2 = -J1_2 / dA
    iJ2_1 = -J2_1 / dA
    iJ2_2 = J1_1 / dA
    # Compute the global derivatives
    D1_1 = dN1_1 * iJ1_1 + dN2_1 * iJ1_2
    D1_2 = dN1_2 * iJ1_1 + dN2_2 * iJ1_2
    D1_3 = dN1_3 * iJ1_1 + dN2_3 * iJ1_2
    D1_4 = dN1_4 * iJ1_1 + dN2_4 * iJ1_2
    D1_5 = dN1_5 * iJ1_1 + dN2_5 * iJ1_2
    D1_6 = dN1_6 * iJ1_1 + dN2_6 * iJ1_2
    D1_7 = dN1_7 * iJ1_1 + dN2_7 * iJ1_2
    D1_8 = dN1_8 * iJ1_1 + dN2_8 * iJ1_2
    D2_1 = dN1_1 * iJ2_1 + dN2_1 * iJ2_2
    D2_2 = dN1_2 * iJ2_1 + dN2_2 * iJ2_2
    D2_3 = dN1_3 * iJ2_1 + dN2_3 * iJ2_2
    D2_4 = dN1_4 * iJ2_1 + dN2_4 * iJ2_2
    D2_5 = dN1_5 * iJ2_1 + dN2_5 * iJ2_2
    D2_6 = dN1_6 * iJ2_1 + dN2_6 * iJ2_2
    D2_7 = dN1_7 * iJ2_1 + dN2_7 * iJ2_2
    D2_8 = dN1_8 * iJ2_1 + dN2_8 * iJ2_2
    # return derivatives
    return D1_1, D1_2, D1_3, D1_4, D1_5, D1_6, D1_7, D1_8, D2_1, D2_2, D2_3, D2_4, D2_5, D2_6, D2_7, D2_8, dA


# function 49: select pile elements routine
def select_pile_elements(free, pile, no, el):
    pn = np.where((no[:, 0] == free[(pile[0] - 1), 0]) & (no[:, 1] >= free[(pile[0] - 1), 1] - pile[1]))[0]
    id = np.where(np.isin(el[:, 0], pn))[0]
    return id


# function 50: solver for the displacement and stability calculation
def displacement_solver_vectorized(SF, no, nc, el, ec, df, phi, psi, coh, E, nu, Fext, Km_d, ad, lim, tol):
    CONV = True  # Variable to check convergence
    # Calculate the Lame parameters (Lambda and Mu) from elasticity modulus and Poisson's ratio
    Lambda, Mu = elasticity_parameters(E, nu)
    # Obtain integration points in local coordinates
    xi, yi = integration_point()
    # Vectorize the nodal coordinates for all elements
    c1_1, c1_2, c2_1, c2_2, c3_1, c3_2, c4_1, c4_2, c5_1, c5_2, c6_1, c6_2, c7_1, c7_2, c8_1, c8_2 = coordinate_array_vectorization(
        no, el)
    # Create a vector of degrees of freedom
    vId = vector_degree_freedom(df)
    # Calculate shear strength parameters with a reduction factor equivalent to SF
    snph, csph, snps, cohr, dt = shear_strength_reduction(SF, phi, psi, coh, E, nu)
    # Initialize arrays for displacement, plastic strain, stress, yield function, and internal forces
    U, evpt1, evpt2, evpt3, evpt4, sigma, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16 = array_declaration_vectorization(
        nc, ec)
    # Solve the displacement field
    U[ad] = Km_d.solve(Fext[ad])
    k = 0
    for k in range(lim):
        # Vectorize the displacements for all nodes
        u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16 = displacement_vectorization(U, df)
        # Loop over each integration point
        for j in range(4):
            id = range(j, 4 * ec, 4)
            # Get the local derivatives for the current integration point
            dN1_1, dN1_2, dN1_3, dN1_4, dN1_5, dN1_6, dN1_7, dN1_8, dN2_1, dN2_2, dN2_3, dN2_4, dN2_5, dN2_6, dN2_7, dN2_8 = local_derivative8(
                xi[j], yi[j])
            # Calculate the global derivatives and area element
            D1_1, D1_2, D1_3, D1_4, D1_5, D1_6, D1_7, D1_8, D2_1, D2_2, D2_3, D2_4, D2_5, D2_6, D2_7, D2_8, dA = global_derivative_vectorized(
                dN1_1, dN1_2, dN1_3, dN1_4, dN1_5, dN1_6, dN1_7, dN1_8, dN2_1, dN2_2, dN2_3, dN2_4, dN2_5, dN2_6, dN2_7,
                dN2_8, c1_1, c1_2, c2_1, c2_2, c3_1, c3_2, c4_1, c4_2, c5_1, c5_2, c6_1, c6_2, c7_1, c7_2, c8_1, c8_2)
            # Calculate strains and stresses for each integration point
            e1, e2, e3, e4 = strain_displacement_vectorized(u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14,
                                                            u15, u16, evpt1[id], evpt2[id], evpt3[id], evpt4[id], D1_1,
                                                            D1_2, D1_3, D1_4, D1_5, D1_6, D1_7, D1_8, D2_1, D2_2, D2_3,
                                                            D2_4, D2_5, D2_6, D2_7, D2_8)
            s1, s2, s3, s4 = stress_strain_vectorized(Lambda[id], Mu[id], e1, e2, e3, e4)
            # Calculate stress invariant and derivatives
            p, q, t, dp1, dp2, dp3, dp4, dJ2_1, dJ2_2, dJ2_3, dJ2_4, dJ3_1, dJ3_2, dJ3_3, dJ3_4 = stress_invariants(s1,
                                                                                                                    s2,
                                                                                                                    s3,
                                                                                                                    s4)
            # calculate Mohr Coulomb yield function and derivaives
            F, dq1, dq2, dq3, sigma_id = yield_mohr_coulomb(p, q, t, snph[id], csph[id], snps[id], cohr[id])
            sigma[id] = sigma_id
            # caclulate viscoplastic strain increment
            evp1 = dt[id] * F * (dq1 * dp1 + dq2 * dJ2_1 + dq3 * dJ3_1)
            evp2 = dt[id] * F * (dq1 * dp2 + dq2 * dJ2_2 + dq3 * dJ3_2)
            evp3 = dt[id] * F * (dq1 * dp3 + dq2 * dJ2_3 + dq3 * dJ3_3)
            evp4 = dt[id] * F * (dq1 * dp4 + dq2 * dJ2_4 + dq3 * dJ3_4)
            # add viscoplastic strain increment to total viscoplastic strain
            evpt1[id] += evp1
            evpt2[id] += evp2
            evpt3[id] += evp3
            evpt4[id] += evp4
            # caclulate viscoplastic stress
            sp1, sp2, sp3, sp4 = stress_strain_vectorized(Lambda[id], Mu[id], evp1, evp2, evp3, evp4)
            # add viscoplastic stress to viscoplastic body loads
            f1 += dA * (D1_1 * sp1 + D2_1 * sp3)
            f2 += dA * (D1_1 * sp3 + D2_1 * sp2)
            f3 += dA * (D1_2 * sp1 + D2_2 * sp3)
            f4 += dA * (D1_2 * sp3 + D2_2 * sp2)
            f5 += dA * (D1_3 * sp1 + D2_3 * sp3)
            f6 += dA * (D1_3 * sp3 + D2_3 * sp2)
            f7 += dA * (D1_4 * sp1 + D2_4 * sp3)
            f8 += dA * (D1_4 * sp3 + D2_4 * sp2)
            f9 += dA * (D1_5 * sp1 + D2_5 * sp3)
            f10 += dA * (D1_5 * sp3 + D2_5 * sp2)
            f11 += dA * (D1_6 * sp1 + D2_6 * sp3)
            f12 += dA * (D1_6 * sp3 + D2_6 * sp2)
            f13 += dA * (D1_7 * sp1 + D2_7 * sp3)
            f14 += dA * (D1_7 * sp3 + D2_7 * sp2)
            f15 += dA * (D1_8 * sp1 + D2_8 * sp3)
            f16 += dA * (D1_8 * sp3 + D2_8 * sp2)
        # assemble viscoplastic load
        values = np.concatenate([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16])
        Fint = np.bincount(vId, weights=values, minlength=2 * nc)
        # add viscoplastic load to external load vector
        loads = Fext + Fint
        # store previous displacement
        U0 = U.copy()
        # solve system of equations to derive new displacement (active degree of freedom only ad)
        U[ad] = Km_d.solve(loads[ad])
        # calculate the error
        er = np.max(np.abs(U - U0)) / np.max(np.abs(U0))
        if er < tol:
            break
    # check whether solver converged
    if k == lim - 1:
        CONV = False
    # return displacement, viscoplastic strain, shear strength and convergence
    return CONV, U, evpt1, evpt2, evpt3, evpt4, sigma


# function 51: find safety factor by executing displacement solvers with several factors of safety
def find_safety_factor(no, nc, el, ec, df, phi, psi, coh, E, nu, Fext, Km_d, ad, lim, tol):
    # Create an array of safety factors from 0.1 to 5.0 with a step of 0.1
    SF = [1]
    i = 0
    U = 0
    evpt1 = 0
    evpt2 = 0
    evpt3 = 0
    evpt4 = 0
    sigma = 0
    # Loop through each safety factor in the SF array
    for i, sf in enumerate(SF):
        # Call the displacement_solver function to calculate displacement and other values
        CONV, U, evpt1, evpt2, evpt3, evpt4, sigma = displacement_solver_vectorized(sf, no, nc, el, ec, df, phi, psi,
                                                                                    coh, E, nu, Fext, Km_d, ad, lim,
                                                                                    tol)
        # Check if the convergence flag is False (meaning the solver did not converge)
        if not CONV:
            break
    # Determine the factor of safety that caused the solver to stop
    factor_of_safety = SF[i]
    # Return the calculated values and the factor of safety
    return U, evpt1, evpt2, evpt3, evpt4, sigma, factor_of_safety


# function 52: display Monte Carlo results
def displayMonteCarlo(normal_flow_rate, FoS):
    # Create a figure for displaying Monte Carlo results
    plt.figure(figsize=(19.2, 10.8))
    plt.suptitle('Monte Carlo results')
    # Create the first subplot for the histogram of normal_flow_rate
    plt.subplot(1, 2, 1)
    plt.hist(normal_flow_rate, edgecolor='black')
    plt.xlabel('Bin', fontsize=14)
    plt.ylabel('Instances', fontsize=14)
    plt.title('|Flow Rate| [m³/s]', fontsize=14)
    # Create the second subplot for the histogram of FoS (Factor of Safety)
    plt.subplot(1, 2, 2)
    plt.hist(FoS, edgecolor='black')
    plt.xlabel('Bin', fontsize=14)
    plt.ylabel('Instances', fontsize=14)
    plt.title('Factor of Safety [-]', fontsize=14)
    # Display the entire figure with both subplots
    plt.show()


# function 53: display single run results
def displayResults(free, wl, pile, def_scaling, equivalent_pile_thickness, anchor_Id, no, nc, el, ec,
                   nodal_water_pressure, lip, FoS, ks, H, U, evpt1, evpt2, evpt3, evpt4, sigma, active_pile):
    # Calculate the pore pressure P
    P = (H - no[:np.max(el[:, :4]) + 1, 1]) * 10.0
    P[P < 0.0] = 0.0
    # Calculate the average pore pressure Pe for each element
    Pe = (P[el[:, 0]] + P[el[:, 1]] + P[el[:, 2]] + P[el[:, 3]]) / 4
    # Calculate a scaling factor for displacements
    scaling_factor = np.round(def_scaling * np.max(no) / np.max(np.abs(U)))
    # Scale the displacements U
    U_scaling = scaling_factor * U
    # Calculate the deformed node positions (no_d)
    no_d = np.zeros_like(no)
    no_d[:, 0] = no[:, 0] + U_scaling[0:2 * nc:2]
    no_d[:, 1] = no[:, 1] + U_scaling[1:2 * nc:2]
    # Calculate vee and normal_flow_rate using flow_darcy function
    vee, flow_rate = flow_darcy(H, no, el, ec, ks)
    # Calculate the norm of the flow rate vector for the entire mesh
    normal_flow_rate = np.round(np.linalg.norm(flow_rate), 6)
    # Modify sigma for active pile integration points
    if active_pile == True:
        id = select_pile_integration_points(free, pile, equivalent_pile_thickness, lip)
        sigma[id] = 0.0
        pileId = select_pile_elements(free, pile, no, el)
    else:
        pileId = []
    # Create a colormap for visualization
    cmap = cm.viridis
    # Calculate kse (average hydraulic conductivity for elements)
    kse = (ks[0::4] + ks[1::4] + ks[2::4] + ks[3::4]) / 4
    # Normalize colors for hydraulic conductivity
    norm = plt.Normalize(np.min(kse), np.max(kse))
    colors0 = cmap(norm(kse))
    sm1 = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm1.set_array([])
    # Calculate sigmae (average shear strength for elements)
    sigmae = (sigma[0::4] + sigma[1::4] + sigma[2::4] + sigma[3::4]) / 4
    # Normalize colors for shear strength
    norm = plt.Normalize(np.min(sigmae), np.max(sigmae))
    colors1 = cmap(norm(sigmae))
    sm2 = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm2.set_array([])
    # Calculate nevpt (norm of evpt1, evpt2, evpt3, evpt4 for nodes)
    nevpt = np.sqrt(evpt1 ** 2 + evpt2 ** 2 + evpt3 ** 2 + evpt4 ** 2)
    # Calculate nevpte (average norm of evpt for elements)
    nevpte = (nevpt[0::4] + nevpt[1::4] + nevpt[2::4] + nevpt[3::4]) / 4
    # Normalize colors for evpt
    norm = plt.Normalize(np.min(nevpte), np.max(nevpte))
    colors2 = cmap(norm(nevpte))
    # Normalize colors for Pe (average pore pressure for elements)
    norm = plt.Normalize(np.min(Pe), np.max(Pe))
    colors3 = cmap(norm(Pe))
    sm3 = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm3.set_array([])
    # figure 1: hydraulic conductivity
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    plt.title(f'Hydraulic Conductivity [m/s], |Flow Rate| {normal_flow_rate} [m3/d]', fontsize=14)
    for i in range(ec):
        indices = el[i, [0, 4, 1, 5, 2, 6, 3, 7]]
        x = no[indices, 0]
        y = no[indices, 1]
        plt.fill(x, y, color=colors0[i], edgecolor='none', alpha=0.7)
    for i in pileId:
        x_coords = no[el[i, [0, 4, 1, 5, 2, 6, 3, 7]], 0]
        y_coords = no[el[i, [0, 4, 1, 5, 2, 6, 3, 7]], 1]
        plt.fill(x_coords, y_coords, 'red')
    if anchor_Id.size == 1:
        plt.scatter(no[anchor_Id, 0], no[anchor_Id, 1], 100, color='white', marker='s', facecolors='none')
    for i in range(wl.shape[0] - 1):
        plt.plot([wl[i, 0], wl[i + 1, 0]], [wl[i, 1], wl[i + 1, 1]], '-b', linewidth=2)
    if np.sum(np.abs(nodal_water_pressure)) > 0:
        plt.quiver(no[:, 0], no[:, 1], -nodal_water_pressure[::2], -nodal_water_pressure[1::2], scale=5,
                   scale_units='xy', angles='xy')
    plt.colorbar(sm1, ax=ax, orientation='vertical')
    plt.xlabel('Width [m]', fontsize=14)
    plt.ylabel('Height [m]', fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    # figure 2: porepressure & Darcy flow
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    plt.title(f'Porepressure [kPa] & Darcy Flow [m/s], |Flow Rate| {normal_flow_rate} [m3/d]', fontsize=14)
    for i in range(ec):
        indices = el[i, [0, 4, 1, 5, 2, 6, 3, 7]]
        x = no[indices, 0]
        y = no[indices, 1]
        plt.fill(x, y, color=colors3[i], edgecolor='none', alpha=0.7)
    for i in pileId:
        x_coords = no[el[i, [0, 4, 1, 5, 2, 6, 3, 7]], 0]
        y_coords = no[el[i, [0, 4, 1, 5, 2, 6, 3, 7]], 1]
        plt.fill(x_coords, y_coords, 'red')
    if anchor_Id.size == 1:
        plt.scatter(no[anchor_Id, 0], no[anchor_Id, 1], 100, color='white', marker='s', facecolors='none')
    if np.sum(np.abs(vee)) > 0:
        plt.quiver(lip[:, 0], lip[:, 1], vee[0, :], vee[1, :], scale_units='xy', angles='xy', alpha=0.3)
    plt.colorbar(sm3, ax=ax, orientation='vertical')
    plt.xlabel('Width [m]', fontsize=14)
    plt.ylabel('Height [m]', fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    # figure 3: shear strength
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    plt.title(f'Shear Strength [kPa], FS {FoS} [-]', fontsize=14)
    for i in range(ec):
        indices = el[i, [0, 4, 1, 5, 2, 6, 3, 7]]
        x = no[indices, 0]
        y = no[indices, 1]
        plt.fill(x, y, color=colors1[i], edgecolor='none', alpha=0.7)
    for i in pileId:
        x_coords = no[el[i, [0, 4, 1, 5, 2, 6, 3, 7]], 0]
        y_coords = no[el[i, [0, 4, 1, 5, 2, 6, 3, 7]], 1]
        plt.fill(x_coords, y_coords, 'red')
    if anchor_Id.size == 1:
        plt.scatter(no[anchor_Id, 0], no[anchor_Id, 1], 100, color='white', marker='s', facecolors='none')
    for i in range(wl.shape[0] - 1):
        plt.plot([wl[i, 0], wl[i + 1, 0]], [wl[i, 1], wl[i + 1, 1]], '-b', linewidth=2)
    if np.sum(np.abs(nodal_water_pressure)) > 0:
        plt.quiver(no[:, 0], no[:, 1], -nodal_water_pressure[::2], -nodal_water_pressure[1::2], scale=5,
                   scale_units='xy', angles='xy')
    plt.colorbar(sm2, ax=ax, orientation='vertical')
    plt.xlabel('Width [m]', fontsize=14)
    plt.ylabel('Height [m]', fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    # figure 4: Viscoplastic Strain
    plt.figure(figsize=(19.2, 10.8))
    plt.title(f'Viscoplastic Strain [-], FS {FoS} [-]', fontsize=14)
    for i in range(ec):
        indices = el[i, [0, 4, 1, 5, 2, 6, 3, 7]]
        x = no[indices, 0]
        y = no[indices, 1]
        plt.fill(x, y, color=colors2[i], edgecolor='none', alpha=0.7)
    plt.xlabel('Width [m]', fontsize=14)
    plt.ylabel('Height [m]', fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    # figure 5: Deformed Mesh
    plt.figure(figsize=(19.2, 10.8))
    plt.title(f'Deformed Mesh, scaled {scaling_factor} times, FS {FoS} [-]', fontsize=14)
    for i in range(ec):
        indices = el[i, [0, 4, 1, 5, 2, 6, 3, 7]]
        x = no_d[indices, 0]
        y = no_d[indices, 1]
        plt.fill(x, y, 'green', edgecolor='black', alpha=0.3)
    for i in pileId:
        x_coords = no_d[el[i, [0, 4, 1, 5, 2, 6, 3, 7]], 0]
        y_coords = no_d[el[i, [0, 4, 1, 5, 2, 6, 3, 7]], 1]
        plt.fill(x_coords, y_coords, 'red')
    if anchor_Id.size == 1:
        plt.scatter(no_d[anchor_Id, 0], no_d[anchor_Id, 1], 100, color='white', marker='s', facecolors='none')
    plt.xlabel('Width [m]', fontsize=14)
    plt.ylabel('Height [m]', fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# function 54: single Monte Carlo iteration (groundwater-displacement-factor of safety)
def single_run(ks_soil, phi_soil, coh_soil, L, free, pile, inter, phi_pile, phi_inter, dilation_factor,
               psi_pile, psi_inter, coh_pile, coh_inter, equivalent_pile_thickness, lip, active_pile,
               active_inter, H0, fh, fs, no, nc, el, ec, df, E, nu, Km_d, ad, lim, tol):
    # Generate random field instances
    # ks, phi, coh = random_field_instance(ks_soil, phi_soil, coh_soil, ec, L)
    ks, phi, coh = uniform_field_instance(ks_soil, phi_soil, coh_soil, ec)
    # Update hydraulic conductivity at integration points
    ks = hydraulic_conductivity_integration_point(free, pile, ks, equivalent_pile_thickness, lip, active_pile)
    # Update hydraulic conductivity at integration points
    phi, psi, coh = plasticity_parameters_integration_point(free, pile, inter, phi, phi_pile, phi_inter,
                                                            dilation_factor, psi_pile, psi_inter, coh, coh_pile,
                                                            coh_inter, equivalent_pile_thickness, lip, active_pile,
                                                            active_inter)
    # form conductivity matrix
    Kc = conductivity_stiffness(no, el, ec, ks)
    # hydraulic solver
    H = hydraulic_solver(H0, Kc, no, el, fh, fs, lim, tol)
    # Darcy flow
    vee, flow_rate = flow_darcy(H, no, el, ec, ks)
    # Calculate the norm of the flow rate vector for the entire mesh
    normal_flow_rate = np.linalg.norm(flow_rate)
    # hydromechanical coupling
    nodal_pore_pressure = hydro_mechanical_coupling(H, no, nc, el, ec, df)
    # add all external forces
    Fext = nodal_self_weight + nodal_water_pressure + nodal_pore_pressure
    # find safety factor
    U, evpt1, evpt2, evpt3, evpt4, sigma, factor_of_safety = find_safety_factor(no, nc, el, ec, df, phi, psi, coh, E,
                                                                                nu, Fext, Km_d, ad, lim, tol)
    # Excluding restricted nodes and boundary nodes
    ad_x = (ad[np.where(ad % 2 == 0)] / 2).astype(int)
    ad_y = (ad[np.where(ad % 2 != 0)] / 2).astype(int)
    # intersection between ad_x and ad_y
    ad_x = np.intersect1d(ad_x, ad_y)
    # take out the free surface nodes
    mask = np.in1d(ad_x, free_surface_node)
    ad_x = ad_x[~mask]
    # collect input coordinates and input features
    input_coords = no[ad_x]
    input_features = np.array([E_soil_input, phi_soil_input, coh_soil_input, ks_soil_input]).reshape([1, 4])
    # collect output features
    displ = np.stack([U[0:2 * nc:2], U[1:2 * nc:2]], axis=1)
    displ_ad = displ[ad_x]
    # TODO: add pore water pressure or ground water flow velocity
    output_features = np.stack([displ_ad[:, 0], displ_ad[:, 1]], axis=1)

    # save input coordinates and input and output features into a npy file
    np.save(f'{sim_name}_{description}_ml_input_coords.npy', input_coords)
    np.save(f'{sim_name}_{description}_ml_input_feature.npy', input_features)
    np.save(f'{sim_name}_{description}_ml_output_feature.npy', output_features)

    # save simulation data and parameter data for the inference
    # uniformly pick 1000 points from the input coordinates
    n_iter = int(input_coords.shape[0] / num_obs)
    data_file_name = f'{sim_name}_{description}_sim.txt'
    sim_data = {'t': [1]}
    for i, _ in enumerate(input_coords[::n_iter, 0]):
        sim_data[f'displ_x_{i}'] = [displ_ad[i, 0]]
        sim_data[f'displ_y_{i}'] = [displ_ad[i, 1]]
        # sim_data[f'pwp_x_{i}'] = [pwp_ad[i, 1]]
        # sim_data[f'pwp_y_{i}'] = [pwp_ad[i, 1]]

    write_dict_to_file(sim_data, data_file_name)

    data_param_name = f'{sim_name}_{description}_param.txt'
    param_data = {'E': [E_soil_input], 'phi': [phi_soil_input], 'coh': [coh_soil_input], 'ks': [ks_soil_input]}
    write_dict_to_file(param_data, data_param_name)

    # return normalized flow, factor of safety, displacement, viscoplastic strain components, permeability and total head
    return normal_flow_rate, factor_of_safety, nodal_pore_pressure, U, evpt1, evpt2, evpt3, evpt4, phi, psi, coh, sigma, ks, H


# function 55: parallel task Monte Carlo Iterations
def parallel_task(i):
    print(f"working on iteration {i + 1}")
    normal_flow_rate_add, factor_of_safety_add, nodal_pore_pressure, U, evpt1, evpt2, evpt3, evpt4, phi, psi, coh, sigma, ks, H = single_run(
        ks_soil, phi_soil, coh_soil, L, free, pile, inter, phi_pile, phi_inter, dilation_factor, psi_pile, psi_inter,
        coh_pile, coh_inter, equivalent_pile_thickness, lip, active_pile, active_inter, H0, fh, fs, no, nc, el, ec, df,
        E, nu, Km_d, ad, lim, tol)
    return normal_flow_rate_add, factor_of_safety_add


# function 56: parallel execution Monte Carlo Iterations
def execute_parallel(mc, num_cores):
    # Create a Pool of workers to parallelize the task
    with Pool(num_cores) as pool:
        results = []
        # Submit tasks asynchronously using apply_async
        for i in range(mc):
            result = pool.apply_async(parallel_task, args=(i,))
            results.append(result)
        # Retrieve results from async tasks
        results = [res.get() for res in results]
    # Extract normal_flow_rate and factor_of_safety from the results
    normal_flow_rate = np.array([result[0] for result in results])
    factor_of_safety = np.array([result[1] for result in results])
    # Return the computed normal_flow_rate and factor_of_safety
    return normal_flow_rate, factor_of_safety


# def interpolate_from_ipts_to_nodes(field):
#     # Obtain integration points in local coordinates
#     xi, yi = integration_point()
#     # Initialize arrays for nodal values
#     interpolated_field = np.zeros([nc, 2])
#     # Iterate over each element in the mesh
#     for i in range(ec):
#         # Iterate over each integration point
#         for j in range(4):
#             # Calculate the index for the hydraulic conductivity at the integration point
#             id = (i * 4) + j
#             # Calculate the shape tensor for the integration point
#             N = shape_tensor4(xi[j], yi[j])
#             interpolated_field[el[i, 0:4], 0] += N.T * field[id ,0]
#             interpolated_field[el[i, 0:4], 1] += N.T * field[id ,1]
#     # Return the interpolated field
#     return interpolated_field


def write_dict_to_file(data, file_name):
    """
    write a python dictionary data into a text file
    """
    with open(file_name, 'w') as f:
        keys = data.keys()
        f.write('# ' + ' '.join(keys) + '\n')
        num = len(data[list(keys)[0]])
        for i in range(num):
            f.write(' '.join([str(data[key][i]) for key in keys]) + '\n')


# %% input parameters from command line

# Variables received from the user

# Extract parameters from sys.argv
# sys.argv[0] is the script name
E_soil_input = float(sys.argv[1])  # [kPa] Young's modulus of soil
phi_soil_input = float(sys.argv[2])  # [deg] friction angle of soil
coh_soil_input = float(sys.argv[3])  # [kPa] cohesion of soil
ks_soil_input = 0.01  # [m/d] hydraulic conductivity of soil
sim_name = sys.argv[4]  # name of the simulation (dike2D)
description = sys.argv[5]  # description of the simulation
num_obs = int(sys.argv[6])  # number of observations for the inference or uncertainty quantification

# %% Geometry Input
free = np.array([
    [0.0, 10.0],
    [10.0, 10.0],
    [14.0, 14.0],
    [18.0, 14.0],
    [30.0, 8.0],
    [50.0, 8.0],
])  # [m] free surface coordinates

wl = np.array([
    [0.0, 12.0],
    [14.0, 12.0],
    [30.0, 7.5],
    [50.0, 7.5],
])  # [m] water level surface coordinates

disc = [1.0, 0.5]  # [m] FEM element size

# %% Soil Input
clx = 1e10  # [m] correlation length x-dir
cly = 1e10  # [m] correlation length y-dir
E_soil = E_soil_input  # [kPa] Young's modulus soil
nu_soil = 0.30  # [-] Poison's ratio soil
gamma_soil = 20.0  # [kN/m3] specific weight soil
phi_soil = [phi_soil_input, phi_soil_input]  # [deg] friction angle [min|max]
coh_soil = [coh_soil_input, 0]  # [kPa] cohesion [mean|standard deviation]
ks_soil = [0.01, 0]  # [m/d] hydraulic conductivity [mean|standard deviation]
dilation_factor = 0.1  # [psi/phi[-]] dilation angle = dilation_factor*friction angle

# %% Pile Input
active_pile = False  # [True|False] activate pile
pile = [3, 6.0]  # [node|depth[m]] pile node and depth
I_pile = 5e-5  # [m4] surface moment of inertia
E_pile = 200e6  # [kPa] Young's modulus pile
nu_pile = 0.20  # [-] Poison's ratio pile
gamma_pile = 78.5  # [kN/m3] specific weight pile
phi_pile = 0.0  # [deg] friction angle pile
psi_pile = 0.0  # [deg] dilation angle pile
coh_pile = 125e3  # [kPa] cohesion pile

# %% Anchor Input
active_anchor = False  # [True|False] activate anchor of the pile
anchor = [12.0, 5.0, 8.0]  # [connection to pile y-axis [m]|base anchor x-axis[m]|base anchor y-axis[m]]
E_anchor = 200e6  # [kPa] Young's modulus anchor
d_anchor = 0.15  # [m] diameter anchor
HoH_anchor = 2.5  # [m] Center on center distance anchor (out of plane direction)

# %% Interface Input
active_inter = False  # [True|False] activate interface
inter = [3, 4, 3.0]  # [node 1|node 2|depth[m]] start interface node, end interface node, depth of the interface
E_inter = 0.5e5  # [kPa] Young's modulus interface
nu_inter = 0.28  # [-] Poison's ratio interface
gamma_inter = 20.0  # [kN/m3] specific weight interface
phi_inter = 18.0  # [deg] friction angle interface
psi_inter = 0.0  # [deg] dilation angle interface
coh_inter = 8.5  # [kPa] cohesion interface

# %% Solver Input
lim = 500  # [-] iteration limit
tol = 1e-4  # [-] iteration tolerance
def_scaling = 0.05  # [-] deformation scaling
num_cores = 18  # [-] Number of Cores to be used

# %% main script:

# Calculate equivalent pile thickness
equivalent_pile_thickness = equivalent_thickness_rectangle(I_pile)
# Find intersection points
intersection_point_list = find_intersection_points(free, wl)
# Partition the domain
X, Y = divide_FEM_domain(free, disc, pile, equivalent_pile_thickness, intersection_point_list, active_pile)
# Generate the mesh
no, nc, el, ec = mesh_generator(X, Y)
# Elevate the mesh
no, free_surface_node = elevate_mesh(free, X, no, el)
# add anchor
anchor_Id, anchorSpringX, anchorSpringY = anchor_spring(free, pile, anchor, E_anchor, d_anchor, HoH_anchor, no,
                                                        active_anchor)
# element degree of freedom array
df = degree_freedom(el, ec)
# Dirichlet boundary conditions
ad, fh, fs = boundary_dirichlet(wl, X, no, nc, el, free_surface_node)
# Neumann boundary conditions
nodal_water_pressure = boundary_neumann(wl, no, nc, free_surface_node)
# integration point location
lip = integration_point_location(no, el, ec)
# # generate random field base
# L = random_field(lip[:, 0], lip[:, 1], clx, cly)
L = None
# elasticity parameters
E, nu = elasticity_parameters_integration_point(free, pile, inter, E_soil, E_pile, E_inter, nu_soil, nu_pile, nu_inter,
                                                equivalent_pile_thickness, ec, lip, active_pile, active_inter)
# displacement stiffness matrix
Km_d = displacement_stiffness(no, nc, el, ec, df, ad, E, nu, anchor_Id, anchorSpringX, anchorSpringY)
# self weight integration point
gamma = specificWeightIntegrationPoint(free, pile, inter, gamma_soil, gamma_pile, gamma_inter,
                                       equivalent_pile_thickness, ec, lip, active_pile, active_inter)
# self weight vector
nodal_self_weight = body_weight(no, nc, el, ec, gamma)
# initial total head
H0 = initial_head(wl, X, no, el, free_surface_node)

if __name__ == '__main__':
    start_time = time.time()
    normal_flow_rate, factor_of_safety, nodal_pore_pressure, U, evpt1, evpt2, evpt3, evpt4, phi, psi, coh, sigma, ks, H = single_run(
        ks_soil, phi_soil, coh_soil, L, free, pile, inter, phi_pile, phi_inter, dilation_factor, psi_pile, psi_inter,
        coh_pile, coh_inter, equivalent_pile_thickness, lip, active_pile, active_inter, H0, fh, fs, no, nc, el, ec, df,
        E, nu, Km_d, ad, lim, tol)
    end_time = time.time()
    print(f"Single run completed: {end_time - start_time} seconds")
