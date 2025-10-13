import numpy as np

# ------------------------------
# 1) Build snapshot matrix and build POD projection or Auto-econder/decoder
# ------------------------------
def build_snapshots_from_list(U_list, normalization=True):
    """Stack an arbitrary list of channels into a (D, T) snapshot matrix and return shape.

    Parameters
    - U_list: list of arrays, each of shape (T, nx, ny), all with identical shapes

    Returns
    - X: (D, T) uncentered snapshot matrix with D = C*nx*ny where C=len(U_list)
    - shape: (nx, ny)
    - channel_bounds: (C, 2) min/max per channel if normalization else None
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
    channel_bounds = np.zeros((n_channels, 2))
    for c in range(n_channels):
        # Extract channel slice for all time steps
        channel_data = X[c*nx*ny:(c+1)*nx*ny, :]
        channel_bounds[c, 0] = channel_data.min()
        channel_bounds[c, 1] = channel_data.max()
    if normalization:
        X = transform(X, channel_bounds)
    return X, (nx, ny), channel_bounds

def transform(X, channel_bounds):
    """Apply min-max normalization to X using channel_bounds.
    - X: (D, T) snapshot matrix to normalize
    - channel_bounds: (C, 2) min/max per channel
    """
    C = channel_bounds.shape[0]
    D, T = X.shape
    channel_size = D // C
    for c in range(C):
        idx_start = c * channel_size
        idx_end = (c + 1) * channel_size
        a = X[idx_start:idx_end, :] - channel_bounds[c, 0]
        b = channel_bounds[c, 1] - channel_bounds[c, 0] + 1e-12
        X[idx_start:idx_end, :] = a / b
    return X

def inverse_transform(X, channel_bounds):
    """Apply inverse min-max normalization to X using channel_bounds.
    
    - X: (D, T) snapshot matrix to unnormalize
    - channel_bounds: (C, 2) min/max per channel
    """
    # X: (D, T), channel_bounds: (C, 2)
    C = channel_bounds.shape[0]
    D, T = X.shape
    # Compute per-channel indices
    channel_size = D // C
    X_inv = np.zeros_like(X)
    for c in range(C):
        idx_start = c * channel_size
        idx_end = (c + 1) * channel_size
        min_c = channel_bounds[c, 0]  # (T,)
        max_c = channel_bounds[c, 1]  # (T,)
        # Broadcast min/max to shape (channel_size, T)
        min_c_b = np.broadcast_to(min_c, (channel_size, T))
        max_c_b = np.broadcast_to(max_c, (channel_size, T))
        # Inverse min-max normalization
        X_inv[idx_start:idx_end, :] = X[idx_start:idx_end, :] * (max_c_b - min_c_b) + min_c_b
    return X_inv

def center_snapshots(X):
    xbar = X.mean(axis=1, keepdims=True)
    Xc = X - xbar
    return Xc, xbar.squeeze()

def pod(Xc, energy=0.999):
    """Compute POD via SVD of centered snapshots Xc = U S V^T.

    Returns
    - U_r: (D, r) spatial modes capturing given energy fraction
    - A: (T, r) time coefficients where A = (S V^T)^T truncated to r
    - Sr: (r,) retained singular values
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
    """Project derivative snapshots [Vx, Vy] onto POD modes to obtain A_dot.

    Parameters
    - U_r: (D, r) POD basis
    - Vx, Vy: arrays of shape (T, nx, ny) for time derivatives per channel

    Returns
    - A_dot: (T, r) modal time derivatives
    """
    V, _ = build_snapshots_from_list([Vx, Vy])
    A_dot = V.T @ U_r
    return A_dot

def build_master_snapshots(files, channels="disp", t_max=None, normalization=True):
    """Build a global snapshot matrix from multiple runs.

    Parameters
    - files: list of .npy file paths
    - channels: which channels to load (see rom_io.load_2d_trajectory_from_file)
    - t_max: optional cap on time steps per run (None means use all)
    - normalization: True or False for channel-wise [0,1] scaling

    Returns
    - X: (D, T_total) concatenated snapshot matrix from all runs
    - shapes: list of (nx, ny) shapes per run
    - X_all: list of (D, T_i) per-run snapshot matrices
    - channel_bounds: (C, 2) min/max per channel if normalization else None
    """
    from rom_io import load_2d_trajectory_from_file  # local import to avoid circular deps
    X_all = []
    shapes = []
    channel_bounds = []
    for f in files:
        U_list = load_2d_trajectory_from_file(f, channels=channels, t_max=t_max)
        X, shape, bounds = build_snapshots_from_list(U_list, normalization=False)
        X_all.append(X)
        shapes.append(shape)
        channel_bounds.append(bounds)
    X_concat = np.concatenate(X_all, axis=1)

    # Compute global channel bounds
    channel_bounds = np.array(channel_bounds)  # shape: (n_runs, n_channels, 2)
    _, n_channels, _ = channel_bounds.shape

    # look for global min/max per channel across all runs
    global_channel_bounds = np.zeros((n_channels, 2))
    global_channel_bounds[:, 0] = channel_bounds[:, :, 0].min(axis=0)  # min over runs
    global_channel_bounds[:, 1] = channel_bounds[:, :, 1].max(axis=0)  # max over runs
    channel_bounds = global_channel_bounds

    if normalization:
        X_all_norm = []
        for Xi, shape in zip(X_all, shapes):
            Xi = transform(Xi, channel_bounds)
            X_all_norm.append(Xi)
        X_concat = np.concatenate(X_all_norm, axis=1)
    return X_concat, shapes, X_all, channel_bounds

def build_master_pod(X_concat, X_all, energy=0.99):
    """Build a global POD basis from multiple runs and return per-run coefficients.

    Parameters
    - X_concat: concatenated snapshot matrix from all runs
    - X_all: list of (D, T_i) per-run snapshot matrices
    - energy: cumulative energy fraction for basis truncation

    Returns
    - U_r: (D, r) common basis, r chosen by energy
    - A_list: list of (T_i, r) POD coefficients per run
    """
    U_r, A_concat, Svals = pod(X_concat, energy=energy)
    A_list = []
    for X in X_all:
        A_i = (X.T @ U_r)
        A_list.append(A_i)
    return U_r, A_list

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
    """Train an autoencoder on snapshots, return enc/dec/latent A/history."""
    try:
        import importlib
        torch = importlib.import_module('torch')
        nn = importlib.import_module('torch.nn')
        optim = importlib.import_module('torch.optim')
        data_utils = importlib.import_module('torch.utils.data')
        DataLoader, TensorDataset = data_utils.DataLoader, data_utils.TensorDataset
    except Exception as e:
        raise ImportError("PyTorch is required for the autoencoder. Install PyTorch to use AE-SINDy features.") from e

    # Define AE models locally to avoid torch dependency at import time
    class Encoder(nn.Module):
        """Simple MLP encoder mapping from full state (D) to latent (latent_dim)."""
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
        """Simple MLP decoder mapping from latent (latent_dim) back to full state (D)."""
        def __init__(self, latent_dim=8, out_dim=0):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(latent_dim, 128), nn.ReLU(),
                nn.Linear(128, 256), nn.ReLU(),
                nn.Linear(256, out_dim)
            )

        def forward(self, z):
            return self.fc(z)
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