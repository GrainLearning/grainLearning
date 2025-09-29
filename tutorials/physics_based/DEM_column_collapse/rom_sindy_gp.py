import numpy as np
import pysindy as ps
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# ----------------------------
# 2 Fit SINDy or GP on POD coeffs. 
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

    model.fit(A, x_dot=A_dot, t=t)
    return model

def fit_sindycp_continuous(A_list, t_list, u_list, poly_deg_state=2, poly_deg_param=1,
                           thresh=0.1, diff="smoothed"):
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
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.2*np.ptp(t_train)+1e-12,
                                       length_scale_bounds=(1e-6, 1e6)) \
             + WhiteKernel(noise_level=1e-10, noise_level_bounds=(1e-12, 1e-3))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, alpha=0.0)
    gp.fit(t_train.reshape(-1,1), y_train)
    y_pred, _ = gp.predict(t_query.reshape(-1,1), return_std=True)
    return y_pred

# ----------------------------
# 3) Rollout and reconstruct
# ----------------------------
def simulate_and_reconstruct(model, U_r, A0, t_eval, xbar=None):
    A_pred = model.simulate(A0, t_eval)
    X_pred = (U_r @ A_pred.T)
    if xbar is not None:
        X_pred += xbar[:, None]
    return X_pred

def simulate_and_reconstruct_derivative(model, U_r, A0, t_eval):
    A_pred = model.simulate(A0, t_eval)
    A_dot_pred = model.predict(A_pred)
    X_pred = (U_r @ A_pred.T)
    V_pred = (U_r @ A_dot_pred.T)
    return X_pred, V_pred

def simulate_and_reconstruct_gp(U_r, A, t_train, t_query, xbar=None):
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

def simulate_and_reconstruct_cp(model, U_r, A0, u, t_eval, xbar=None):
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
