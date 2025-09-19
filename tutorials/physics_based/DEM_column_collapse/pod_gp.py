import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split

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

def simulate_and_reconstruct_gp(U_r, xbar, A, t_train, t_query):
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

    Xc_pred = (U_r @ A_pred.T)      # (D x T)
    X_pred  = Xc_pred + xbar[:, None]
    return X_pred

def unpack_velocity(xvec, shape):
    nx, ny = shape
    ux = xvec[:nx*ny].reshape(nx, ny)
    uy = xvec[nx*ny:].reshape(nx, ny)
    return ux, uy

def visualize_velocity_magnitude(X_true, X_pred, shape, time_index, tag):
    ux_t, uy_t = unpack_velocity(X_true[:, time_index], shape)
    ux_p, uy_p = unpack_velocity(X_pred[:, time_index], shape)
    sp_true = np.hypot(ux_t, uy_t)
    sp_pred = np.hypot(ux_p, uy_p)

    fig, axs = plt.subplots(1,2, figsize=(8,4), constrained_layout=True)
    im0 = axs[0].imshow(sp_true.T, origin='lower'); axs[0].set_title('speed (true)')
    im1 = axs[1].imshow(sp_pred.T, origin='lower'); axs[1].set_title(f'speed ({tag})')
    fig.colorbar(im0, ax=axs[0], fraction=0.046); fig.colorbar(im1, ax=axs[1], fraction=0.046)
    plt.savefig(f"velocity_magnitude_comparison_{tag}_{time_index}.png")
    plt.close(fig)

def rel_errors_over_time(X_true, X_pred):
    T = X_true.shape[1]
    errs = []
    for k in range(T):
        num = np.linalg.norm(X_true[:, k] - X_pred[:, k])
        den = np.linalg.norm(X_true[:, k]) + 1e-12
        errs.append(num/den)
    return np.array(errs)

# ---------- 5) End-to-end ----------
def main():
    # Load (your file maps “rho/phi” here; switch to vectors['vel'] if you want velocity)
    output = np.load("collumn_collapse_CG_fields.npy", allow_pickle=True).item()
    time_steps = list(output.keys())
    # Choose the channels you want to compress:
    # Example (as in your snippet): use rho & phi as two “channels”
    Ux = np.array([output[k]['scalars']['rho'] for k in time_steps])  # (T, nx, ny)
    Uy = np.array([output[k]['scalars']['phi'] for k in time_steps])  # (T, nx, ny)
    # For velocity instead, uncomment:
    # Ux = np.array([output[k]['vectors']['vel'][0] for k in time_steps])
    # Uy = np.array([output[k]['vectors']['vel'][1] for k in time_steps])
    # For stress instead, uncomment:
    # Ux = np.array([output[k]['tensors']['stress'][0][0] for k in time_steps])  # (T, nx, ny)
    # Uy = np.array([output[k]['tensors']['stress'][1][1] for k in time_steps])  # (T, nx, ny)

    # DEM base dt times an index stride (assuming uniform integer time_steps)
    dt = 1.97e-5 * (time_steps[1] - time_steps[0])

    # Build snapshots & POD (with proper centering and mean add-back later)
    X, shape = build_snapshots(Ux, Uy)
    Xc, xbar = center_snapshots(X)
    U_r, A, Svals = pod(Xc, energy=0.99)
    r_full = U_r.shape[1]
    print(f"[POD] kept r = {r_full} modes (energy 99%)")

    # Simulate with Gaussian Process (sklearn)
    num_modes = min(10, r_full)
    T = A.shape[0]
    t_query = np.arange(T) * dt
    X_pred_gp = simulate_and_reconstruct_gp(U_r[:, :num_modes], xbar, A[:, :num_modes], t_query, t_query)
    relB = np.linalg.norm(X - X_pred_gp) / (np.linalg.norm(X) + 1e-12)
    errsB = rel_errors_over_time(X, X_pred_gp)
    print(f"[POD+GP] global relative error: {relB:.4f}")
    for k in range(len(errsB)):
        print(f" Step {k:4d}: rel. err = {errsB[k]:.4f}")

    visualize_velocity_magnitude(X, X_pred_gp, shape, time_index=T//2, tag=f"GP")
    
    errors = rel_errors_over_time(X, X_pred_gp)
    for k in range(len(errors)):
        print(f" Step {k:4d}: rel. err = {errors[k]:.4f}")

    # Select every nth snapshot for train/test split
    # n = 5  # e.g., select every 5th snapshot
    # Xc_train = Xc[:, ::n]
    # t_train = t_query[::n]

    # Select the first half for training
    mid = Xc.shape[1] // 2
    Xc_train = Xc[:, :mid]
    t_train = t_query[:mid]

    # POD on training set
    U_r_train, A_train, Svals_train = pod(Xc_train, energy=0.99)
    r_train = U_r_train.shape[1]
    print(f"[POD] train: kept r = {r_train} modes (energy 99%)")

    # Fit GP on training POD coefficients
    num_modes = min(10, r_train)
    T = A_train.shape[0]
    X_pred_gp = simulate_and_reconstruct_gp(U_r_train[:, :num_modes], xbar, A_train[:, :num_modes], t_train, t_query)
    relB = np.linalg.norm(X - X_pred_gp) / (np.linalg.norm(X) + 1e-12)
    errsB = rel_errors_over_time(X, X_pred_gp)
    print(f"[POD+GP] global relative error: {relB:.4f}")
    for k in range(len(errsB)):
        print(f" Step {k:4d}: rel. err = {errsB[k]:.4f}")

    # Visualize one test snapshot
    visualize_velocity_magnitude(X, X_pred_gp, shape, time_index=T//2, tag=f"GP_train")

if __name__ == "__main__":
    main()
