import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
import torch
import torch.nn as nn
import torch.optim as optim
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Autoencoder definition ---
class Encoder(nn.Module):
    def __init__(self, in_dim, latent_dim=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=8, out_dim=0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, z):
        return self.fc(z)

def train_autoencoder(
    X_snapshots,              # (D, T)
    latent_dim=8,
    epochs=50,
    batch_size=32,
    lr=1e-3,
    device="cpu",
    val_split=0.2,            # <-- NEW: hold out % of time steps for validation
    patience=100,             # <-- NEW: early stopping on val loss
    print_every=10,
    seed=0
):
    """
    Trains AE with a time-wise train/val split.
    Returns: enc, dec, A (T x latent_dim), history (dict)
    - Best weights (lowest val loss) are restored before encoding all T frames.
    """
    import math
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    rng = np.random.RandomState(seed)

    D, T = X_snapshots.shape
    X_all = X_snapshots.T.astype(np.float32)   # (T, D)
    idx = np.arange(T)
    rng.shuffle(idx)

    # time-wise split (keeps leakage low for rollout tasks)
    n_val = int(round(val_split * T)) if 0 < val_split < 1 else int(val_split)
    n_val = max(1, min(T-1, n_val)) if T > 1 else 0
    val_idx = np.sort(idx[:n_val])
    trn_idx = np.sort(idx[n_val:]) if n_val > 0 else np.sort(idx)

    X_trn = torch.tensor(X_all[trn_idx], dtype=torch.float32)
    X_val = torch.tensor(X_all[val_idx], dtype=torch.float32) if n_val > 0 else None
    X_all_t = torch.tensor(X_all, dtype=torch.float32)  # for final encoding

    trn_loader = DataLoader(TensorDataset(X_trn), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False, drop_last=False) if X_val is not None else None

    # Models
    enc = Encoder(D, latent_dim).to(device)
    dec = Decoder(latent_dim, D).to(device)
    opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    crit = nn.MSELoss()

    best_state = None
    best_val = math.inf
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "best_val": None, "trn_idx": trn_idx, "val_idx": val_idx}

    for ep in range(1, epochs+1):
        # ---- Train
        enc.train(); dec.train()
        trn_sum, n_trn = 0.0, 0
        for (xb,) in trn_loader:
            xb = xb.to(device)
            z  = enc(xb)
            xr = dec(z)
            loss = crit(xr, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = xb.size(0)
            trn_sum += loss.item() * bs
            n_trn   += bs
        trn_loss = trn_sum / max(1, n_trn)

        # ---- Val
        if val_loader is not None:
            enc.eval(); dec.eval()
            with torch.no_grad():
                val_sum, n_val_batches = 0.0, 0
                for (xb,) in val_loader:
                    xb = xb.to(device)
                    z  = enc(xb)
                    xr = dec(z)
                    val_sum += crit(xr, xb).item() * xb.size(0)
                    n_val_batches += xb.size(0)
            val_loss = val_sum / max(1, n_val_batches)
        else:
            val_loss = trn_loss  # no val set

        history["train_loss"].append(trn_loss)
        history["val_loss"].append(val_loss)

        if ep % print_every == 0 or ep == 1:
            print(f"[AE] epoch {ep}/{epochs}  train={trn_loss:.6e}  val={val_loss:.6e}  (best={best_val:.6e})")

        # ---- Early stopping on validation
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            history["best_val"] = best_val
            no_improve = 0
            # Save best weights
            best_state = {
                "enc": {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                "dec": {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()},
            }
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[AE] Early stopping at epoch {ep} (no val improvement for {patience} epochs).")
                break

    # ---- Restore best weights
    if best_state is not None:
        enc.load_state_dict(best_state["enc"])
        dec.load_state_dict(best_state["dec"])

    # ---- Encode ALL frames with best model
    enc.eval()
    with torch.no_grad():
        A = enc(X_all_t.to(device)).cpu().numpy()

    return enc, dec, A, history

def plot_ae_history(history, savepath="ae_loss_curve.png"):
    """
    Plot train vs val loss from history returned by train_autoencoder.
    """
    if "train_loss" not in history or "val_loss" not in history:
        print("[plot_ae_history] No losses in history dict.")
        return

    plt.figure(figsize=(6,4))
    plt.plot(history["train_loss"], label="train", lw=2)
    plt.plot(history["val_loss"], label="val", lw=2)
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
    print(f"[plot_ae_history] Saved to {savepath}")

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

def simulate_and_reconstruct_autoencoder(model, dec, A0, t_eval, xbar=None, integrator="solve_ivp", device="cpu"):
    A_pred = model.simulate(A0, t_eval,
                            integrator=integrator,
                            integrator_kws=dict(rtol=1e-8, atol=1e-10)  # optional tolerances
                            )  # (T, latent_dim) 
    A_pred_torch = torch.tensor(A_pred, dtype=torch.float32).to(device)
    with torch.no_grad():
        X_pred = dec(A_pred_torch).cpu().numpy().T  # (D, T)
    if xbar is not None:
        X_pred += xbar[:, None]
    return X_pred  # (D, T)

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
    time_steps = sorted(output.keys())
    # Ux = np.array([output[k]['scalars']['rho'] for k in time_steps])  # (T, nx, ny)
    # Uy = np.array([output[k]['scalars']['phi'] for k in time_steps])  # (T, nx, ny)
    Ux = np.array([output[k]['vectors']['disp'][0] for k in time_steps], dtype=np.float64)
    Uy = np.array([output[k]['vectors']['disp'][1] for k in time_steps], dtype=np.float64)
    # Ux = np.array([output[k]['vectors']['vel'][0]  for k in time_steps])  # (T, nx, ny)
    # Uy = np.array([output[k]['vectors']['vel'][1]  for k in time_steps])  # (T, nx, ny)

    # Ux = np.array([output[k]['tensors']['stress'][0][0] for k in time_steps])  # (T, nx, ny)
    # Uy = np.array([output[k]['tensors']['stress'][1][1] for k in time_steps])  # (T, nx, ny)
    
    # DEM timestep size
    dt = 1.97e-5
    dt *= (time_steps[1] - time_steps[0])

    # Build centered snapshots and POD or Autoencoder
    X, shape = build_snapshots(Ux, Uy)
    epochs = 5000
    enc, dec, A, hist = train_autoencoder(X, latent_dim=2, epochs=epochs, batch_size=64, lr=5e-4, device="cuda", val_split=0.2, patience=200, print_every=50)
    # Save and check the loss curves
    plot_ae_history(hist, savepath="ae_train_val_loss.png")

    # Fit SINDy
    t = np.arange(A.shape[0]) * dt
    model = fit_sindy_continuous(A, t, poly_degree=2, thresh=1, diff="smoothed")
    model.print()

    # Rollout from first state for the whole horizon
    A0 = A[0, :]
    X_pred = simulate_and_reconstruct_autoencoder(model, dec, A0, t_eval=t, device="cuda")

    # Error metrics
    print_error_metrics(X, X_pred)

    # Visualize the 2D field over time (10 snapshots)
    for i in range(0, len(t), len(t)//10):
        visualize_2d_field_magnitude(X, X_pred, shape, time_index=i, name='vec_field')

    # # Select every nth snapshot for train/test split
    # n = 5  # e.g., select every 5th snapshot
    # X_train = X[:, ::n]
    # t_train = t[::n]

    # Select the first half for training
    mid = X.shape[1] // 2
    X_train = X[:, :mid]
    t_train = t[:mid]

    # POD on training set
    enc, dec, A_train, hist = train_autoencoder(X_train, latent_dim=2, epochs=epochs, batch_size=64, lr=5e-4, device="cuda", val_split=0.2, patience=200, print_every=50)
    plot_ae_history(hist, savepath="test_ae_train_val_loss.png")

    # Fit SINDy on training POD coefficients
    model_train = fit_sindy_continuous(A_train, t_train, poly_degree=2, thresh=1, diff="smoothed")
    model_train.print()

    # Rollout on test set
    A0_test = A_train[0, :]
    X_test_pred = simulate_and_reconstruct_autoencoder(model_train, dec, A0_test, t_eval=t,
                                                       integrator="solve_ivp", device="cuda")
    # decrease the polynomial degree if the rollout is unstable
    while X_test_pred.shape != X.shape:
        print(f"Warning: test prediction shape {X_test_pred.shape} does not match original {X.shape}")
        model_train = fit_sindy_continuous(A_train, t_train, poly_degree=1, thresh=1, diff="smoothed")
        model_train.print()
        X_test_pred = simulate_and_reconstruct_autoencoder(model_train, dec, A0_test, t_eval=t,
                                                           integrator="solve_ivp", device="cuda")

    # Error metrics on test set
    print_error_metrics(X, X_test_pred)

    # Visualize the 2D field over time (10 snapshots)
    for i in range(0, len(t), len(t)//10):
        visualize_2d_field_magnitude(X, X_test_pred, shape, time_index=i, name='test_vec_field')

if __name__ == "__main__":
    main()
