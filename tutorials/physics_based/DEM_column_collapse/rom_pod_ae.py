import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------
# 1) Build snapshot matrix and build POD projection or Auto-econder/decoder
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

def build_snapshots_from_list(U_list):
    """
    Stack channels in the list into column snapshots.
    Returns X (D,T) uncentered and shape=(nx,ny)
    """
    n_channels = len(U_list)
    assert len(U_list) >= 1
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
    SVD of (centered) snapshots: Xc = U S V^T
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
    V, _ = build_snapshots(Vx, Vy)
    A_dot = V.T @ U_r
    return A_dot

def build_master_pod(files, channels="disp", energy=0.99, t_max=-1):
    """
    Build a single POD basis from concatenated snapshots over all parameter runs.
    Returns:
      U_r: (D,r) common basis, r from energy
      A_list: list of (T_i x r) POD coefficients per run
      shapes, dts: ancillary info
    """
    from rom_io import load_2d_trajectory_from_file
    X_all = []
    shapes = []
    traj_X = []
    for f in files:
        X, shape = load_2d_trajectory_from_file(f, channels=channels, t_max=t_max)
        X_all.append(X)
        traj_X.append(X)
        shapes.append(shape)
    X_concat = np.concatenate(X_all, axis=1)
    U_r, A_concat, Svals = pod(X_concat, energy=energy)

    A_list = []
    for X in traj_X:
        A_i = (X.T @ U_r)
        A_list.append(A_i)
    return U_r, A_list, traj_X, shapes

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
    X_snapshots,
    latent_dim=8,
    epochs=50,
    batch_size=32,
    lr=1e-3,
    device="cpu",
    val_split=0.2,
    patience=100,
    print_every=10,
    seed=0
):
    import math
    rng = np.random.RandomState(seed)

    D, T = X_snapshots.shape
    X_all = X_snapshots.T.astype(np.float32)
    idx = np.arange(T)
    rng.shuffle(idx)

    n_val = int(round(val_split * T)) if 0 < val_split < 1 else int(val_split)
    n_val = max(1, min(T-1, n_val)) if T > 1 else 0
    val_idx = np.sort(idx[:n_val])
    trn_idx = np.sort(idx[n_val:]) if n_val > 0 else np.sort(idx)

    X_trn = torch.tensor(X_all[trn_idx], dtype=torch.float32)
    X_val = torch.tensor(X_all[val_idx], dtype=torch.float32) if n_val > 0 else None
    X_all_t = torch.tensor(X_all, dtype=torch.float32)

    trn_loader = DataLoader(TensorDataset(X_trn), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False, drop_last=False) if X_val is not None else None

    enc = Encoder(D, latent_dim).to(device)
    dec = Decoder(latent_dim, D).to(device)
    opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    crit = nn.MSELoss()

    best_state = None
    best_val = math.inf
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "best_val": None, "trn_idx": trn_idx, "val_idx": val_idx}

    for ep in range(1, epochs+1):
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
            val_loss = trn_loss

        history["train_loss"].append(trn_loss)
        history["val_loss"].append(val_loss)

        if ep % print_every == 0 or ep == 1:
            print(f"[AE] epoch {ep}/{epochs}  train={trn_loss:.6e}  val={val_loss:.6e}  (best={best_val:.6e})")

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            history["best_val"] = best_val
            no_improve = 0
            best_state = {
                "enc": {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                "dec": {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()},
            }
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[AE] Early stopping at epoch {ep} (no val improvement for {patience} epochs).")
                break

    if best_state is not None:
        enc.load_state_dict(best_state["enc"])
        dec.load_state_dict(best_state["dec"])

    enc.eval()
    with torch.no_grad():
        A = enc(X_all_t.to(device)).cpu().numpy()

    return enc, dec, A, history
