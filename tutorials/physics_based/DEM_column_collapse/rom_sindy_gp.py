import numpy as np
import pysindy as ps
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# ----------------------------
# 2 Fit SINDy or GP on POD coeffs. 
# ----------------------------
def fit_sindy_continuous(A, t, poly_degree=3, thresh=0.5, diff="finite"):
    """Fit a continuous-time SINDy model a'(t) = f(a) from sampled coefficients.

    - A: (T, r) POD (or latent) coefficients over time.
    - t: (T,) time vector.
    - poly_degree, thresh, diff: library/optimizer and differentiation options.
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

    model.fit(A, x_dot=A_dot, t=t)
    return model

def fit_sindycp_continuous(A_list, t_list, u_list, poly_deg_state=2, poly_deg_param=1,
                           thresh=0.1, diff="smoothed"):
    """Fit a parameterized SINDy (SINDy-CP) model a'(t) = f(a, u).

    - A_list: list of (T_i, r) trajectories.
    - t_list: list/array of times (uniform assumed by PySINDy when given scalars).
    - u_list: list of parameters per trajectory (vector or time series).
    """
    if diff == "finite":
        diff_method = ps.FiniteDifference()
    elif diff == "smoothed":
        diff_method = ps.SmoothedFiniteDifference()
    else:
        raise ValueError("diff must be 'finite' or 'smoothed'")

    feature_lib   = ps.PolynomialLibrary(degree=poly_deg_state, include_bias=True)
    parameter_lib = ps.PolynomialLibrary(degree=poly_deg_param, include_bias=True)

    r = A_list[0].shape[1]
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

    model.fit(A_list, u=u_list, t=t_list)
    return model

def fit_predict_gp_sklearn(t_train, y_train, t_query):
    """Fit a 1D GP on (t_train, y_train) and predict at t_query using RBF+white kernel.

    The kernel form is RBF + WhiteKernel; its hyperparameters are optimized via
    scikit-learn's GaussianProcessRegressor (with restarts).
    """
    # The hyperparameters (kernel parameters) are automatically optimized by GaussianProcessRegressor during fitting.
    # You can access the optimized parameters via gp.kernel_ after fitting.
    kernel = C(1.0, (1e-4, 1e4)) * RBF(length_scale=0.2*np.ptp(t_train)+1e-12,
                                       length_scale_bounds=(1e-8, 1e8)) \
             + WhiteKernel(noise_level=1e-10, noise_level_bounds=(1e-12, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, alpha=0.0)
    gp.fit(t_train.reshape(-1,1), y_train)
    # Optimized kernel parameters are in gp.kernel_
    y_pred, _ = gp.predict(t_query.reshape(-1,1), return_std=True)
    return y_pred

class multivariate_GP:
    """
    Multivariable GP to map parameters to outputs via independent per-mode GPs.
    For example, Learns A0(u): R^m -> R^r
    """
    def __init__(self, kernels=None, n_restarts=3, normalize_y=True, random_state=0):
        # list of kernels per output dim, or None to auto-build
        self.kernels = kernels
        self.n_restarts = n_restarts
        self.normalize_y = normalize_y
        self.random_state = random_state
        self.scaler_u = StandardScaler()
        self.scaler_y = StandardScaler()
        # list[GaussianProcessRegressor]
        self.gps = []
        self.r = None
        self.m = None

    def fit(self, U_list, A_list):
        """
        U_list: list of parameter vectors u_i (shape (m,))
        A_list: list of output.
            e.g., POD coeff trajectories A^(i) (T_i x r); we use A0 = A^(i)[0]
        """
        U = np.vstack([np.atleast_2d(u) for u in U_list])    # (N, m)
        Y = np.vstack([A_list])       # (N, r)

        self.m = U.shape[1]
        self.r = Y.shape[1]

        # scale inputs and outputs
        Uz = self.scaler_u.fit_transform(U)
        Yz = self.scaler_y.fit_transform(Y)

        # build default kernels if not provided
        if self.kernels is None:
            # reasonable default: σ^2 * RBF(ℓ) + white noise
            self.kernels = [
                C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
                + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-2))
                for _ in range(self.r)
            ]

        self.gps = []
        for j in range(self.r):
            gp = GaussianProcessRegressor(
                kernel=self.kernels[j],
                n_restarts_optimizer=self.n_restarts,
                normalize_y=self.normalize_y,
                random_state=self.random_state,
                alpha=0.0,
            )
            gp.fit(Uz, Yz[:, j])
            self.gps.append(gp)
        return self

    def predict(self, u_new, return_std=False):
        """
        u_new: (m,) or (N, m)
        Returns output: (N, r), and optionally std: (N, r) in original scale.
        """
        Uq = np.atleast_2d(u_new)
        Uqz = self.scaler_u.transform(Uq)

        Yz_pred = np.zeros((Uqz.shape[0], self.r))
        Yz_std  = np.zeros_like(Yz_pred) if return_std else None

        for j, gp in enumerate(self.gps):
            if return_std:
                mu_j, std_j = gp.predict(Uqz, return_std=True)
                Yz_pred[:, j] = mu_j
                Yz_std[:, j]  = std_j
            else:
                Yz_pred[:, j] = gp.predict(Uqz, return_std=False)

        # invert output scaling
        Y_pred = self.scaler_y.inverse_transform(Yz_pred)
        if return_std:
            # std needs to be scaled by output std only (no mean shift)
            scale = self.scaler_y.scale_[None, :]
            return Y_pred, Yz_std * scale
        return Y_pred

# ----------------------------
# 3) Rollout and reconstruct
# ----------------------------
def simulate_and_reconstruct(model, U_r, A0, t_eval, xbar=None):
    """Roll out SINDy in coefficient space and reconstruct field with POD modes."""
    A_pred = model.simulate(A0, t_eval)
    X_pred = (U_r @ A_pred.T)
    if xbar is not None:
        X_pred += xbar[:, None]
    return X_pred

def simulate_and_reconstruct_derivative(model, U_r, A0, t_eval):
    """Roll out SINDy and also return reconstructed time derivative via model.predict."""
    A_pred = model.simulate(A0, t_eval)
    A_dot_pred = model.predict(A_pred)
    X_pred = (U_r @ A_pred.T)
    V_pred = (U_r @ A_dot_pred.T)
    return X_pred, V_pred

def simulate_and_reconstruct_gp(U_r, A, t_train, t_query, xbar=None):
    """Fit independent GPs a_j(t) and reconstruct X at t_query using the POD modes."""
    T, r = A.shape
    A_pred = np.zeros((t_query.shape[0], r))
    for j in range(r):
        y = A[:, j]
        yhat = fit_predict_gp_sklearn(t_train, y, t_query)
        A_pred[:, j] = yhat

    X_pred = (U_r @ A_pred.T)
    if xbar is not None:
        X_pred  = X_pred + xbar[:, None]
    return X_pred

def simulate_and_reconstruct_autoencoder(model, dec, A0, t_eval, xbar=None, integrator="solve_ivp", device="cpu"):
    """Roll out SINDy in AE latent space and decode to full field using 'dec'."""
    import torch
    A_pred = model.simulate(A0, t_eval,
                            integrator=integrator,
                            integrator_kws=dict(rtol=1e-8, atol=1e-10)
                            )
    A_pred_torch = torch.tensor(A_pred, dtype=torch.float32).to(device)
    with torch.no_grad():
        X_pred = dec(A_pred_torch).cpu().numpy().T
    if xbar is not None:
        X_pred += xbar[:, None]
    return X_pred

def simulate_and_reconstruct_cp(model, U_r, A0, t_eval, u, xbar=None):
    """Simulate SINDy-CP with constant or time-varying parameters u and reconstruct X."""
    if np.ndim(u) == 1:
        def u_fun(tau, u_const=u):
            tau = np.atleast_1d(tau)
            return np.tile(u_const, (tau.size, 1))
        A_pred = model.simulate(A0, t_eval, u=u_fun)
    else:
        assert u.shape[0] == t_eval.size, "u time length must match t_eval"
        A_pred = model.simulate(A0, t_eval, u=u)
    X_pred = (U_r @ A_pred.T)
    if xbar is not None:
        X_pred += xbar[:, None]
    return X_pred
