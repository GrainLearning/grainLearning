import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

def build_snapshots(Ux, Uy):
    """
    Stack channels [Ux, Uy] into column snapshots.
    Returns X (D,T) uncentered and shape=(nx,ny)
    """
    T, nx, ny = Ux.shape
    D = 2 * nx * ny
    X = np.empty((D, T))
    for k in range(T):
        xk = np.concatenate([Ux[k].ravel(), Uy[k].ravel()])
        X[:, k] = xk
    return X, (nx, ny)

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

def center_snapshots(X):
    xbar = X.mean(axis=1, keepdims=True)
    Xc = X - xbar
    return Xc, xbar.squeeze()

def pod(Xc, energy=0.999):
    """
    SVD of centered snapshots: Xc = U S V^T
    Returns U_r (D,r), A (T,r), Svals[:r]
    """
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    cum = np.cumsum(S**2) / np.sum(S**2)
    r = np.searchsorted(cum, energy) + 1
    U_r = U[:, :r]
    Sr  = S[:r]
    Vt_r= Vt[:r, :]
    A   = (Sr[:, None] * Vt_r).T   # (T x r)
    return U_r, A, Sr

def _fit_predict_gp_sklearn(t_train, y_train, t_query):
    # Kernel: σ^2 * RBF(ℓ) + white noise
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
        yhat = _fit_predict_gp_sklearn(t_train, y, t_query)
        A_pred[:, j] = yhat

    X_pred = (U_r @ A_pred.T)      # (D x T)
    if xbar is not None:
        X_pred  = X_pred + xbar[:, None]
    return X_pred

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

def print_error_metrics(X, X_pred):
    rel = np.linalg.norm(X - X_pred) / np.linalg.norm(X)
    print(f"Global relative error: {rel:.4f}")
    for k in range(X.shape[1]):
        relk = np.linalg.norm(X[:, k] - X_pred[:, k]) / np.linalg.norm(X[:, k])
        print(f"  step {k:4d} relative error: {relk:.4f}")

# ---------- 5) End-to-end ----------
def main():
    # Load (your file maps “rho/phi” here; switch to vectors['vel'] if you want velocity)
    output = np.load("collumn_collapse_CG_fields.npy", allow_pickle=True).item()
    time_steps = list(output.keys())
    # Choose the channels you want to compress:
    # Example (as in your snippet): use rho & phi as two “channels”
    Occ = np.array([output[k]['scalars']['occ'] for k in time_steps])  # (T, nx, ny)
    # Uy = np.array([output[k]['scalars']['phi'] for k in time_steps])  # (T, nx, ny)
    # For velocity instead, uncomment:
    # Ux = np.array([output[k]['vectors']['vel'][0] for k in time_steps])
    # Uy = np.array([output[k]['vectors']['vel'][1] for k in time_steps])
    Ux = np.array([output[k]['vectors']['disp'][0] for k in time_steps])
    Uy = np.array([output[k]['vectors']['disp'][1] for k in time_steps])
    # For stress instead, uncomment:
    # Ux = np.array([output[k]['tensors']['stress'][0][0] for k in time_steps])  # (T, nx, ny)
    # Uy = np.array([output[k]['tensors']['stress'][1][1] for k in time_steps])  # (T, nx, ny)

    # DEM base dt times an index stride (assuming uniform integer time_steps)
    dt = 1.97e-5 * (time_steps[1] - time_steps[0])

    # Build snapshots & POD (with proper centering and mean add-back later)
    X, shape = build_snapshots_from_list([Occ, Ux, Uy])
    Xc, xbar = center_snapshots(X)
    U_r, A, Svals = pod(Xc, energy=0.99)
    r_full = U_r.shape[1]
    print(f"[POD] kept r = {r_full} modes (energy 99%)")

    # Simulate with Gaussian Process (sklearn)
    num_modes = min(10, r_full)
    T = A.shape[0]
    t_query = np.arange(T) * dt
    X_pred_gp = simulate_and_reconstruct_gp(U_r[:, :num_modes], A[:, :num_modes], t_query, t_query, xbar=xbar)

    # Error metrics
    print_error_metrics(X, X_pred_gp)

    # Visualize the 2D field over time (10 snapshots)
    for i in range(0, len(t_query), len(t_query)//10):
        visualize_2d_field_magnitude(X, X_pred_gp, shape, time_index=i, channels=[1, 2], name='vel_field_magnitude')
        visualize_2d_field(X, X_pred_gp, shape, time_index=i, channel=0, name='occ_field')

    # Select every nth snapshot for train/test split
    # n = 5  # e.g., select every 5th snapshot
    # X_train = X[:, ::n]
    # t_train = t_query[::n]

    # Select the first half for training
    mid = X.shape[1] // 2
    X_train = X[:, :mid]
    t_train = t_query[:mid]

    # POD on training set
    Xc_train, xbar_train = center_snapshots(X_train)
    U_r_train, A_train, Svals_train = pod(Xc_train, energy=0.99)
    r_train = U_r_train.shape[1]
    print(f"[POD] train: kept r = {r_train} modes (energy 99%)")

    # Fit GP on training POD coefficients
    num_modes = min(10, r_train)
    T = A_train.shape[0]
    X_pred_gp = simulate_and_reconstruct_gp(U_r_train[:, :num_modes], A_train[:, :num_modes], t_train, t_query, xbar=xbar_train)

    # Error metrics on test set
    print_error_metrics(X, X_pred_gp)

    # Visualize the 2D field over time (10 snapshots)
    for i in range(0, len(t_query), len(t_query)//10):
        visualize_2d_field_magnitude(X, X_pred_gp, shape, time_index=i, channels=[1, 2], name='test_vel_field_magnitude')
        visualize_2d_field(X, X_pred_gp, shape, time_index=i, channel=0, name='test_occ_field')


if __name__ == "__main__":
    main()
