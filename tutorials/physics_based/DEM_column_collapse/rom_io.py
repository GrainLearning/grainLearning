import glob, re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------------------
# 0) Helper functions for input and output
# ----------------------------------------
def load_2d_trajectory_from_file(npy_path, channels="disp", t_max=-1):
    """
    Load (Ux,Uy) from a single npy file written by your CG pipeline.
      channels='disp' | 'vel' | 'scalar_rho_phi' etc.
    Returns X (D, T) and shape info from build_snapshots.
    """
    from rom_pod_ae import build_snapshots  # local import to avoid circular deps
    print(f"[load_2d_trajectory_from_file] Loading {npy_path} with channels={channels} until t_max={t_max}")
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
    plt.close()

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
    plt.close()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def create_gif_from_pngs(name='2d_field'):
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
