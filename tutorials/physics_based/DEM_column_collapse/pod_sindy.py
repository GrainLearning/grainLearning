import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from sklearn.model_selection import train_test_split

# ----------------------------
# 1) Build snapshot matrix & POD
# ----------------------------
def build_snapshots(Ux, Uy):
    """
    Stack channels [Ux, Uy] into column snapshots.
    Returns X (D, T) and mean field xbar (D,)
    """
    T, nx, ny = Ux.shape
    D = 2 * nx * ny
    X = np.empty((D, T))
    for k in range(T):
        xk = np.concatenate([Ux[k].ravel(), Uy[k].ravel()])
        X[:, k] = xk
    return X, (nx, ny)

def build_snapshots_without_mean(Ux, Uy):
    """
    Stack channels [Ux, Uy] into column snapshots.
    Returns X (D, T) and mean field xbar (D,)
    """
    T, nx, ny = Ux.shape
    D = 2 * nx * ny
    X = np.empty((D, T))
    for k in range(T):
        xk = np.concatenate([Ux[k].ravel(), Uy[k].ravel()])
        X[:, k] = xk
    xbar = X.mean(axis=1, keepdims=True)
    Xc = X - xbar
    return Xc, xbar.squeeze(), (nx, ny)

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

def project_data_to_modal_derivative(U_r, Vx, Vy):
    V, _ = build_snapshots(Vx, Vy)   # V is (D,T) derivative snapshots (e.g. velocity)
    A_dot = V.T @ U_r
    return A_dot

# ----------------------------
# 2) Fit SINDy on POD coeffs
# ----------------------------
def fit_sindy_continuous(A, t, poly_degree=3, thresh=0.5, diff="finite"):
    """
    Learn \dot a = f(a) using SINDy (continuous-time ODE).
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

def unpack_2d_field(xvec, shape):
    nx, ny = shape
    ux = xvec[:nx*ny].reshape(nx, ny)
    uy = xvec[nx*ny:].reshape(nx, ny)
    return ux, uy

def visualize_2d_field_magnitude(X, X_pred, shape, time_index, name='2d_field'):
    ux_t, uy_t = unpack_2d_field(X[:, time_index], shape)
    ux_p, uy_p = unpack_2d_field(X_pred[:, time_index], shape)
    sp_true = np.hypot(ux_t, uy_t)
    sp_pred = np.hypot(ux_p, uy_p)

    fig, axs = plt.subplots(1,2, figsize=(8,4), constrained_layout=True)
    im0 = axs[0].imshow(sp_true.T, origin='lower'); axs[0].set_title('speed (true)')
    im1 = axs[1].imshow(sp_pred.T, origin='lower'); axs[1].set_title('speed (SINDy)')
    fig.colorbar(im0, ax=axs[0], fraction=0.046); fig.colorbar(im1, ax=axs[1], fraction=0.046)
    plt.savefig(f"{name}_at_{time_index}.png")

def print_error_metrics(X, X_pred):
    rel = np.linalg.norm(X - X_pred) / np.linalg.norm(X)
    print(f"Global relative error: {rel:.4f}")
    for k in range(X.shape[1]):
        relk = np.linalg.norm(X[:, k] - X_pred[:, k]) / np.linalg.norm(X[:, k])
        print(f"  step {k:4d} relative error: {relk:.4f}")

# ----------------------------
# 4) End-to-end
# ----------------------------
def main():
    # Load or replace with your data loader
    output = np.load("collumn_collapse_CG_fields.npy", allow_pickle=True).item()
    time_steps = list(output.keys())
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
    X, shape = build_snapshots(Ux, Uy)
    U_r, A, Svals = pod(X, energy=0.99)
    r = U_r.shape[1]
    print(f"[POD] kept r = {r} modes (energy 99%)")

    # Fit SINDy
    num_modes = min(10, r)
    t = np.arange(A.shape[0]) * dt
    model = fit_sindy_continuous(A[:, :num_modes], t, poly_degree=2, thresh=1, diff="smoothed")
    model.print()

    # Rollout from first state for the whole horizon
    A0 = A[0, :num_modes]
    X_pred = simulate_and_reconstruct(model, U_r[:, :num_modes], A0, t_eval=t)

    # Error metrics
    print_error_metrics(X, X_pred)

    # Visualize the 2D field over time (10 snapshots)
    for i in range(0, len(t), len(t)//10):
        visualize_2d_field_magnitude(X, X_pred, shape, time_index=i, name='vec_field')

    # Select every nth snapshot for train/test split
    # n = 5  # e.g., select every 5th snapshot
    # X_train = X[:, ::n]
    # t_train = t[::n]

    # Select the first half for training
    mid = X.shape[1] // 2
    X_train = X[:, :mid]
    t_train = t[:mid]

    # POD on training set
    U_r_train, A_train, Svals_train = pod(X_train, energy=0.99)
    r_train = U_r_train.shape[1]
    print(f"[POD] train: kept r = {r_train} modes (energy 99%)")

    # Fit SINDy on training POD coefficients
    num_modes_train = min(2, r_train)
    model_train = fit_sindy_continuous(A_train[:, :num_modes_train], t_train, poly_degree=2, thresh=1, diff="smoothed")
    model_train.print()

    # Rollout on test set
    A0_test = A_train[0, :num_modes_train]
    X_test_pred = simulate_and_reconstruct(model_train, U_r_train[:, :num_modes_train], A0_test, t_eval=t)

    # Error metrics on test set
    print_error_metrics(X, X_test_pred)

    # Visualize the 2D field over time (10 snapshots)
    for i in range(0, len(t), len(t)//10):
        visualize_2d_field_magnitude(X, X_pred, shape, time_index=i, name='test_vec_field')

if __name__ == "__main__":
    main()
