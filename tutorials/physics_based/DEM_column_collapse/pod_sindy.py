import glob, re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pysindy as ps
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# ----------------------------------------
# 0) Helper functions for input and output
# ----------------------------------------
def load_2d_trajectory_from_file(npy_path, channels="disp", t_max=-1):
    """
    Load (Ux,Uy) from a single npy file written by your CG pipeline.
      channels='disp' | 'vel' | 'scalar_rho_phi' etc.
    Returns Ux, Uy, shape info is in build_snapshots.
    """
    out = np.load(npy_path, allow_pickle=True).item()
    time_steps = list(out.keys())
    time_steps = time_steps[:t_max]
    if channels == "disp":
        Ux = np.array([out[k].item()['vectors']['disp'][0] for k in time_steps])
        Uy = np.array([out[k].item()['vectors']['disp'][1] for k in time_steps])
    elif channels == "vel":
        Ux = np.array([out[k].item()['vectors']['vel'][0]  for k in time_steps])
        Uy = np.array([out[k].item()['vectors']['vel'][1]  for k in time_steps])
    elif channels == "scalar_rho_phi":
        Ux = np.array([out[k].item()['scalars']['rho'] for k in time_steps])
        Uy = np.array([out[k].item()['scalars']['phi'] for k in time_steps])
    else:
        raise ValueError("Unsupported channels")
    X, shape = build_snapshots(Ux, Uy)
    return X, shape

def unpack_2d_field(xvec, shape, channels):
    nx, ny = shape
    fields = []
    for c in channels:
        fields.append(xvec[c * nx * ny:(c + 1) * nx * ny].reshape(nx, ny))
    return fields

def visualize_2d_field_magnitude(X, X_pred, shape, time_index, channels=[0, 1], name='2d_field'):
    # unpack the channels
    fields = unpack_2d_field(X[:, time_index], shape, channels)
    fields_pred = unpack_2d_field(X_pred[:, time_index], shape, channels)
    # compute the magnitude
    if len(fields) != 2 or len(fields_pred) != 2:
        raise ValueError("channels must be of length 2 for magnitude visualization")
    sp_true = np.hypot(fields[0], fields[1])
    sp_pred = np.hypot(fields_pred[0], fields_pred[1])
    # plot side-by-side true, pred, and relative error
    fig, axs = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
    im0 = axs[0].imshow(sp_true.T, origin='lower'); axs[0].set_title('speed (true)')
    im1 = axs[1].imshow(sp_pred.T, origin='lower'); axs[1].set_title('speed (GP)')
    # error calculation should avoid elements where true is zero
    valid = np.where(sp_true.T > 0)
    error = np.zeros_like(sp_true.T)
    error[valid] = np.abs(sp_true.T[valid] - sp_pred.T[valid]) / sp_true.T[valid]
    im2 = axs[2].imshow(error, origin='lower', vmin=0, vmax=1); axs[2].set_title('relative error')
    fig.colorbar(im0, ax=axs[0], fraction=0.046); fig.colorbar(im1, ax=axs[1], fraction=0.046); fig.colorbar(im2, ax=axs[2], fraction=0.046)
    plt.savefig(f"{name}_at_{time_index}.png")

def visualize_2d_field(X, X_pred, shape, time_index, channel=0, name='2d_field'):
    # unpack the channel
    fields = unpack_2d_field(X[:, time_index], shape, [channel])
    fields_pred = unpack_2d_field(X_pred[:, time_index], shape, [channel])
    # plot side-by-side true, pred, and relative error
    fig, axs = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
    im0 = axs[0].imshow(fields[0].T, origin='lower'); axs[0].set_title('field (true)')
    im1 = axs[1].imshow(fields_pred[0].T, origin='lower'); axs[1].set_title('field (GP)')
    # error calculation should avoid elements where true is zero
    valid = np.where(fields[0].T > 0)
    error = np.zeros_like(fields[0].T)
    error[valid] = np.abs(fields[0].T[valid] - fields_pred[0].T[valid]) / fields[0].T[valid]
    im2 = axs[2].imshow(error, origin='lower', vmin=0, vmax=1); axs[2].set_title('relative error')
    fig.colorbar(im0, ax=axs[0], fraction=0.046); fig.colorbar(im1, ax=axs[1], fraction=0.046); fig.colorbar(im2, ax=axs[2], fraction=0.046)
    plt.savefig(f"{name}_at_{time_index}.png")

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def create_gif_from_pngs(X, X_pred, shape, every=10, name='2d_field'):
    for i in range(0, X.shape[1], every):
        visualize_2d_field_magnitude(X, X_pred, shape, time_index=i, name=name)
    png_files = sorted(glob.glob(f"{name}_at_*.png"), key=natural_sort_key)
    frames = [Image.open(p) for p in png_files]
    # Save as GIF
    frames[0].save(
        f"{name}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=200,  # ms per frame
        loop=0         # infinite loop
    )

# ------------------------------
# 1) Build snapshot matrix & POD
# ------------------------------
def build_snapshots(Ux, Uy):
    """
    Stack channels [Ux, Uy] into column snapshots.
    Returns X (D, T) and shape=(nx, ny).
    """
    T, nx, ny = Ux.shape
    D = 2 * nx * ny
    X = np.empty((D, T))
    for k in range(T):
        xk = np.concatenate([Ux[k].ravel(), Uy[k].ravel()])
        X[:, k] = xk
    return X, (nx, ny)

def center_snapshots(X):
    xbar = X.mean(axis=1, keepdims=True)
    Xc = X - xbar
    return Xc, xbar.squeeze()

def build_snapshots_from_list(U_list):
    """
    Stack channels in the list into column snapshots.
    Returns X (D,T) uncentered and shape=(nx,ny)
    """
    # assert the length of U_list is at least 1
    n_channels = len(U_list)
    assert len(U_list) >= 1
    # assert all elements in U_list have the same shape
    shapes = [u.shape for u in U_list]
    assert all(s == shapes[0] for s in shapes), "All elements in U_list must have the same shape" 
    T, nx, ny = shapes[0]
    D = n_channels * nx * ny
    X = np.empty((D, T))
    for k in range(T):
        xk = np.concatenate([u[k].ravel() for u in U_list])
        X[:, k] = xk
    return X, (nx, ny)

def pod(Xc, energy=0.999):
    """
    SVD of centered snapshots: Xc = U S V^T
    - U: spatial modes (D x r)
    - A: time coefficients (T x r) given by A = (S V^T)^T
    """
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    cum = np.cumsum(S**2) / np.sum(S**2)
    r = np.searchsorted(cum, energy) + 1
    U_r = U[:, :r]
    Sr  = S[:r]
    Vt_r= Vt[:r, :]
    A   = (Sr[:, None] * Vt_r).T   # (T x r)
    return U_r, A, Sr

def build_master_pod(files, channels="disp", energy=0.99, t_max=-1):
    """
    Build a single POD basis from concatenated snapshots over all parameter runs.
    Returns:
      U_r: (D,r) common basis, r from energy
      A_list: list of (T_i x r) POD coefficients per run
      shapes, dts: ancillary info
    """
    X_all = []
    shapes = []
    traj_X = []   # keep each run's X for later recon/metrics
    for f in files:
        X, shape = load_2d_trajectory_from_file(f, channels=channels, t_max=t_max)
        X_all.append(X)
        traj_X.append(X)
        shapes.append(shape)
    # Concatenate along time to build a master basis
    X_concat = np.concatenate(X_all, axis=1)  # (D, sum T_i)
    U_r, A_concat, Svals = pod(X_concat, energy=energy)

    # Split A_concat back into per-trajectory A
    A_list = []
    for X in traj_X:
        # Project this run's X onto the master basis via the same SVD factors:
        A_i = (X.T @ U_r)
        A_list.append(A_i)
    return U_r, A_list, traj_X, shapes

def project_data_to_modal_derivative(U_r, Vx, Vy):
    V, _ = build_snapshots(Vx, Vy)   # V is (D,T) derivative snapshots (e.g. velocity)
    A_dot = V.T @ U_r
    return A_dot

# ----------------------------
# 2.1) Fit SINDy on POD coeffs
# ----------------------------
def fit_sindy_continuous(A, t, poly_degree=3, thresh=0.5, diff="finite"):
    """
    Learn dynamics a'(t) = f(a(t)) using SINDy (continuous-time ODE).
    """
    if diff == "finite":
        diff_method = ps.FiniteDifference()
    elif diff == "smoothed":
        diff_method = ps.SmoothedFiniteDifference()
    else:
        raise ValueError("diff must be 'finite' or 'smoothed'")

    lib = ps.PolynomialLibrary(degree=poly_degree, include_interaction=True, include_bias=True)
    opt = ps.STLSQ(threshold=thresh, alpha=1e-6, normalize_columns=True)

    model = ps.SINDy(feature_library=lib,
                     optimizer=opt,
                     differentiation_method=diff_method)
    model.fit(A, t=t)
    return model

def fit_sindy_with_derivative(A, A_dot, t, poly_degree=3, thresh=0.5):
    lib = ps.PolynomialLibrary(degree=poly_degree, include_interaction=True, include_bias=True)
    opt = ps.STLSQ(threshold=thresh, alpha=1e-6, normalize_columns=True)

    model = ps.SINDy(feature_library=lib,
                     optimizer=opt)

    model.fit(A, x_dot=A_dot, t=t)  # <-- use provided derivatives
    return model

def fit_sindycp_continuous(A_list, t_list, u_list, poly_deg_state=2, poly_deg_param=1,
                           thresh=0.1, diff="smoothed"):
    """
    Learn parameterized dynamics  a'(t) = f(a(t), u)
    Inputs:
      - A_list:   list of (T_i x r) POD coefficient arrays for each trajectory
      - t_list:   list of scalar timesteps
      - u_list:   list of controls (parameters) per trajectory:
                   * constant vector per traj: e.g., [mu1, mu2, ...]
                   * OR time series u_i(t): shape (T_i, m)
      - poly_deg_state: degree for state library
      - poly_deg_param: degree for parameter library
      - thresh: sparsity threshold for STLSQ
      - diff: 'finite' or 'smoothed' for derivative estimation
      - feature_names_state: optional names for state features
      - feature_names_param: optional names for parameter features
    Returns:
      - trained ps.SINDy model with a ParameterizedLibrary
    """
    if diff == "finite":
        diff_method = ps.FiniteDifference()
    elif diff == "smoothed":
        diff_method = ps.SmoothedFiniteDifference()
    else:
        raise ValueError("diff must be 'finite' or 'smoothed'")

    # Libraries: separate for state and parameters (as in SINDyCP docs)
    feature_lib   = ps.PolynomialLibrary(degree=poly_deg_state, include_bias=True)
    parameter_lib = ps.PolynomialLibrary(degree=poly_deg_param, include_bias=True)

    # Determine sizes
    r = A_list[0].shape[1]
    # If u_list entries are vectors, infer m from length; if arrays, from last dim
    if np.ndim(u_list[0]) == 1:
        m = len(u_list[0])
    else:
        m = u_list[0].shape[1]

    lib = ps.ParameterizedLibrary(
        feature_library=feature_lib,
        parameter_library=parameter_lib,
        num_features=r,
        num_parameters=m,
    )

    opt = ps.STLSQ(threshold=thresh, normalize_columns=True)
    model = ps.SINDy(
        feature_library=lib,
        optimizer=opt,
        differentiation_method=diff_method,
    )

    # PySINDy SINDyCP expects multiple_trajectories=True with u_list aligned
    model.fit(A_list, u=u_list, t=t_list)
    return model

# -------------------------
# 2.2) Fit GP on POD coeffs
# -------------------------
def fit_predict_gp_sklearn(t_train, y_train, t_query):
    # Kernel: \sigma^2 * RBF(\tau) + white noise
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.2*np.ptp(t_train)+1e-12,
                                       length_scale_bounds=(1e-6, 1e6)) \
             + WhiteKernel(noise_level=1e-10, noise_level_bounds=(1e-12, 1e-3))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, alpha=0.0)
    gp.fit(t_train.reshape(-1,1), y_train)
    y_pred, _ = gp.predict(t_query.reshape(-1,1), return_std=True)
    return y_pred

def simulate_and_reconstruct_gp(U_r, A, t_train, t_query, xbar=None):
    """
    Fit independent GPs for each mode coefficient a_j(t).
    Return X_pred over the same timeline as A.
    """
    T, r = A.shape
    A_pred = np.zeros((t_query.shape[0], r))
    for j in range(r):
        y = A[:, j]
        yhat = fit_predict_gp_sklearn(t_train, y, t_query)
        A_pred[:, j] = yhat

    X_pred = (U_r @ A_pred.T)      # (D x T)
    if xbar is not None:
        X_pred  = X_pred + xbar[:, None]
    return X_pred

# ----------------------------
# 3) Rollout and reconstruct
# ----------------------------
def simulate_and_reconstruct(model, U_r, A0, t_eval, xbar=None):
    A_pred = model.simulate(A0, t_eval)
    X_pred = (U_r @ A_pred.T)     # (D x T)
    if xbar is not None:
        X_pred += xbar[:, None]
    return X_pred                 # (T x D)

def simulate_and_reconstruct_derivative(model, U_r, A0, t_eval):
    A_pred = model.simulate(A0, t_eval)
    A_dot_pred = model.predict(A_pred)
    X_pred = (U_r @ A_pred.T)     # (D x T)
    V_pred = (U_r @ A_dot_pred.T) # (D x T)
    return X_pred, V_pred         # (T x D)

def simulate_and_reconstruct_cp(model, U_r, A0, u, t_eval, xbar=None):
    """Handles constant or time-varying u"""
    if np.ndim(u) == 1:
        # constant controls -> pass a callable
        def u_fun(tau, u_const=u):
            tau = np.atleast_1d(tau)
            return np.tile(u_const, (tau.size, 1))
        A_pred = model.simulate(A0, t_eval, u=u_fun)
    else:
        # time-varying controls
        assert u.shape[0] == t_eval.size, "u time length must match t_eval"
        A_pred = model.simulate(A0, t_eval, u=u)
    X_pred = (U_r @ A_pred.T)     # (D x T)
    if xbar is not None:
        X_pred += xbar[:, None]
    return X_pred                 # (T x D)

def print_error_metrics(X, X_pred, tag=""):
    print_global_error(X, X_pred, tag=tag)    
    for k in range(X.shape[1]):
        num = np.linalg.norm(X[:, k] - X_pred[:, k])
        den = np.linalg.norm(X[:, k]) + 1e-12
        relk = num / den
        print(f"  step {k:4d} relative error: {relk:.4f}")

def print_global_error(X, X_pred, tag=""):
    rel = np.linalg.norm(X - X_pred) / (np.linalg.norm(X) + 1e-12)
    print(f"{tag} Global relative error: {rel:.4f}")
    return rel

# ----------------------------
# 4) End-to-end
# ----------------------------
def ROM_time():
    # Load or replace with your data loader
    output = np.load("collumn_collapse_CG_fields.npy", allow_pickle=True).item()
    time_steps = list(output.keys())
    Occ = np.array([output[k]['scalars']['occ'] for k in time_steps])  # (T, nx, ny)
    # Ux = np.array([output[k]['scalars']['rho'] for k in time_steps])  # (T, nx, ny)
    # Uy = np.array([output[k]['scalars']['phi'] for k in time_steps])  # (T, nx, ny)
    Ux = np.array([output[k]['vectors']['disp'][0] for k in time_steps])  # (T, nx, ny)
    Uy = np.array([output[k]['vectors']['disp'][1] for k in time_steps])  # (T, nx, ny)
    # Ux = np.array([output[k]['vectors']['vel'][0]  for k in time_steps])  # (T, nx, ny)
    # Uy = np.array([output[k]['vectors']['vel'][1]  for k in time_steps])  # (T, nx, ny)

    # Ux = np.array([output[k]['tensors']['stress'][0][0] for k in time_steps])  # (T, nx, ny)
    # Uy = np.array([output[k]['tensors']['stress'][1][1] for k in time_steps])  # (T, nx, ny)
    
    # DEM timestep size
    dt = 1.97e-5
    dt *= (time_steps[1] - time_steps[0])

    # Build centered snapshots and POD
    X, shape = build_snapshots_from_list([Occ, Ux, Uy])
    Xc, xbar = center_snapshots(X)
    U_r, A, Svals = pod(Xc, energy=0.99)
    r = U_r.shape[1]
    print(f"[POD] kept r = {r} modes (energy 99%)")

    # Fit SINDy
    num_modes = min(3, r)
    t = np.arange(A.shape[0]) * dt
    model = fit_sindy_continuous(A[:, :num_modes], t, poly_degree=1, thresh=0.1, diff="smoothed")
    model.print()

    # Rollout from first state for the whole horizon
    A0 = A[0, :num_modes]
    X_pred = simulate_and_reconstruct(model, U_r[:, :num_modes], A0, t_eval=t, xbar=xbar)

    # Error metrics
    print_error_metrics(X, X_pred)

    # Visualize the 2D field over time (10 snapshots)
    for i in range(0, len(t), len(t)//10):
        visualize_2d_field_magnitude(X, X_pred, shape, time_index=i, channels=[1, 2], name='vel_field_magnitude')
        visualize_2d_field(X, X_pred, shape, time_index=i, channel=0, name='occ_field')

    # Select every nth snapshot for train/test split
    # n = 5  # e.g., select every 5th snapshot
    # X_train = X[:, ::n]
    # t_train = t[::n]

    # Select the first half for training
    mid = X.shape[1] // 2
    X_train = X[:, :mid]
    t_train = t[:mid]

    # Center training data
    Xc_train, xbar_train = center_snapshots(X_train)
        
    # POD on training set
    U_r_train, A_train, Svals_train = pod(Xc_train, energy=0.99)
    r_train = U_r_train.shape[1]
    print(f"[POD] train: kept r = {r_train} modes (energy 99%)")

    # Fit SINDy on training POD coefficients
    num_modes_train = min(3, r_train)
    model_train = fit_sindy_continuous(A_train[:, :num_modes_train], t_train, poly_degree=1, thresh=0.1, diff="smoothed")
    model_train.print()

    # Rollout on test set
    A0_test = A_train[0, :num_modes_train]
    X_test_pred = simulate_and_reconstruct(model_train, U_r_train[:, :num_modes_train], A0_test, t_eval=t, xbar=xbar_train)

    # Error metrics on test set
    print_error_metrics(X, X_test_pred)

    # Visualize the 2D field over time (10 snapshots)
    for i in range(0, len(t), len(t)//10):
        visualize_2d_field_magnitude(X, X_test_pred, shape, time_index=i, channels=[1, 2], name='test_vel_field_magnitude')
        visualize_2d_field(X, X_test_pred, shape, time_index=i, channel=0, name='test_occ_field')

def POD_SINDy_parametric_time(file_list, u_list, t_list, channels="disp", energy=0.99, num_modes=10, t_max=-1):
    """
    End-to-end parametric ROM with POD + SINDy-CP.
    Inputs:
      - file_list: list of npy files with CG fields for different parameter settings
      - u_list: list of parameter vectors (or time series) corresponding to file_list
      - t_list: list of time arrays corresponding to file_list
      - channels: which CG fields to use ('disp', 'vel', 'scalar_rho_phi', etc)
      - energy: POD energy threshold for master basis
      - num_modes: number of POD modes to keep for SINDy-CP training
      - t_max: max number of time steps to load from each file (-1 for all)
    """
    # --------- Build a common POD basis and per-run coefficients ----------
    U_r, A_list_full, X_list, shapes = build_master_pod(file_list, channels=channels, energy=energy, t_max=t_max)
    r_full = U_r.shape[1]
    print(f"[POD] train: kept r = {r_full} modes (energy {energy*100:.1f}%)")

    # Truncate modes consistently across runs
    r_use = min(num_modes, r_full)
    A_list = [A[:, :r_use] for A in A_list_full]
    U_use  = U_r[:, :r_use]

    # --------- Fit SINDy-CP (separate polynomial libraries for state and parameters) ----------
    model = fit_sindycp_continuous(
        A_list=A_list,
        t_list=t_list,
        u_list=u_list,
        poly_deg_state=1,
        poly_deg_param=1,
        thresh=0.1,
        diff="smoothed"
    )
    model.print()
    return model, U_use, A_list, X_list, shapes

def evaluate_parametric_rom(model, U_use, A0_list, X_list, shapes, u_list, t_list, channels="disp", create_visual=False, every=10):
    avg_error = 0.0
    for i, (A0, X, shape) in enumerate(zip(A0_list, X_list, shapes)):
        u  = u_list[i]
        X_pred = simulate_and_reconstruct_cp(model, U_use, A0, t_eval=t_list, u=u)
        tag = f"[run {i}]"
        avg_error += print_global_error(X, X_pred, tag=tag)
        # Save evolution of the 2D field into a GIF
        if create_visual:
            create_gif_from_pngs(X, X_pred, shape, name=f"{channels}_Sample{i:02d}", every=every)
    print(f"Average error across all runs: {avg_error / len(A_list):.4f}")

def ROM_parametric_time():
    import glob, os
    # Load datasets
    data_dir = 'Yade/column_collapse_sim_data/'
    # Get list of data files for different parameter settings
    file_list = glob.glob(os.path.join(data_dir, "column_collapse_*_CG_fields.npy"))
    # Load parameter values from file skipping the first row and the first two columns
    param_file = os.path.join(data_dir, "collapse_Iter0_Samples.txt")
    param_data = np.loadtxt(param_file, skiprows=1, usecols=range(3,6))
    u_list = list([data for data in param_data])
    # Check if the number of files matches the number of parameter sets
    assert len(file_list) == param_data.shape[0], "Number of data files and parameter sets do not match"
    # Extract time steps from the first file
    output0 = np.load(file_list[0], allow_pickle=True).item()
    dt = 1e3 * 1.97e-5
    t_max = 200
    t_list = (np.array(list(output0.keys())) + 1e4) * dt
    t_list = t_list[:t_max]
    # Prepare lists for trajectories
    channels = "disp"
    energy = 0.99    # retained energy for master POD
    num_modes = 10    # truncation for SINDy-CP training
    # Build and evaluate the parametric ROM
    model, U_use, A_list, X_list, shapes = POD_SINDy_parametric_time(file_list, u_list, t_list, channels=channels, energy=energy, num_modes=num_modes, t_max=t_max)
    A0_list = [A[0] for A in A_list]
    evaluate_parametric_rom(model, U_use, A0_list, X_list, shapes, u_list, t_list, channels=channels, create_visual=True, every=10)

    # --------- Split data into train/test runs and evaluate ----------
    mid = len(file_list) // 2
    file_list_train = file_list[:mid]
    file_list_test = file_list[mid:]
    u_list_train = u_list[:mid]
    u_list_test = u_list[mid:]
    # Build a GP to predict cofficients at time 0 for test runs
    model_train, U_use_train, A_list_train, X_list_train, shapes_train = POD_SINDy_parametric_time(
        file_list_train, u_list_train, t_list, channels=channels, energy=energy, num_modes=num_modes, t_max=t_max)
    A0_train = np.array([A[0] for A in A_list_train])
    A0_test = fit_predict_gp_sklearn(u_list_train, A0_train, u_list_test)
    
    # Evaluate on test runs
    evaluate_parametric_rom(model_train, U_use_train, A_list_test, X_list_test, shapes_test, u_list_test, t_list, channels=channels, create_visual=False)

if __name__ == "__main__":
    ROM_parametric_time()
