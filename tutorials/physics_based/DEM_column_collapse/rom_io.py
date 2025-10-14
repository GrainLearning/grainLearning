import os, glob, re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------------------
# 0) Helper functions for input and output
# ----------------------------------------
def load_2d_trajectory_from_file(npy_path, channels=["rho"], t_max=None):
    """Load a 2D trajectory and stack requested channels into a list.

    Parameters
    - npy_path: Path to a .npy file produced by the CG pipeline (dict keyed by time step index).
    - channels: Which data to read, one or a combination of 
      ['disp_x', 'disp_y', 'vel_x', 'vel_y', 'rho', 'occ'].
      ['disp_x', 'disp_y'] read a 2D vector field; ['rho', 'occ'] reads two scalar fields.
    - t_max: Optional cap on number of time steps to load (useful for quick tests).

    Returns
    - U_list: list of arrays, each of shape (T, nx, ny), all with identical shapes
    """
    print(f"[load_2d_trajectory_from_file] Loading {npy_path} with channels={channels} until t_max={t_max}")
    out = np.load(npy_path, allow_pickle=True).item()
    time_steps = list(out.keys())
    time_steps = time_steps[:t_max]
    U_list = []
    for ch in channels:
        if ch == "disp_x":
            U_list.append(np.array([out[k].item()['vectors']['disp'][0] for k in time_steps]))
        elif ch == "disp_y":
            U_list.append(np.array([out[k].item()['vectors']['disp'][1] for k in time_steps]))
        elif ch == "vel_x":
            U_list.append(np.array([out[k].item()['vectors']['vel'][0]  for k in time_steps]))
        elif ch == "vel_y":
            U_list.append(np.array([out[k].item()['vectors']['vel'][1]  for k in time_steps]))
        elif ch == "rho":
            U_list.append(np.array([out[k].item()['scalars']['rho'] for k in time_steps]))
        elif ch == "occ":
            U_list.append(np.array([out[k].item()['scalars']['occ'] for k in time_steps]))
        else:
            raise ValueError(f"Unsupported channel {ch}")
    return U_list

def unpack_2d_field(xvec, shape, channels):
    """Slice flattened multi-channel vector into selected channel 2D arrays.

    Parameters
    - xvec: 1D array of length D = C*nx*ny, flattened stacked channels.
    - shape: (nx, ny) original grid size.
    - channels: list of integer channel indices to extract (0-based).

    Returns
    - List of 2D arrays, each of shape (nx, ny), in the same order as 'channels'.
    """
    nx, ny = shape
    fields = []
    for c in channels:
        fields.append(xvec[c * nx * ny:(c + 1) * nx * ny].reshape(nx, ny))
    return fields

def visualize_2d_field_magnitude(X, X_pred, shape, time_index, channels=[0, 1], name='2d_field', tag=""):
    """Save side-by-side magnitude plots (true, predicted, relative error) at a time index.

    - X, X_pred: (D, T) matrices (flattened stacked channels), same D and T.
    - shape: (nx, ny) grid shape.
    - time_index: integer time column to visualize.
    - channels: two channel indices whose magnitude is computed via hypot.
    - name: filename stem; file saved as '{name}_at_{time_index}.png'.
    """
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
    im0 = axs[0].imshow(sp_true.T, origin='lower'); axs[0].set_title(f'field mag. ({tag})')
    im1 = axs[1].imshow(sp_pred.T, origin='lower', vmin=im0.get_clim()[0], vmax=im0.get_clim()[1]); axs[1].set_title(f'field mag. ({tag})')
    # error calculation should avoid elements where true is zero
    valid = np.where(sp_true.T != 0)
    error = np.zeros_like(sp_true.T)
    error[valid] = np.abs(sp_true.T[valid] - sp_pred.T[valid]) / sp_true.T[valid]
    im2 = axs[2].imshow(error, origin='lower', vmin=0, vmax=1); axs[2].set_title('relative error')
    fig.colorbar(im0, ax=axs[0], fraction=0.046); fig.colorbar(im1, ax=axs[1], fraction=0.046); fig.colorbar(im2, ax=axs[2], fraction=0.046)
    plt.savefig(f"{tag}_{name}_at_{time_index}.png")
    plt.close()

def visualize_2d_field(X, X_pred, shape, time_index, channel=0, name='2d_field', tag=""):
    """Save side-by-side scalar field plots (true, predicted, relative error) for one channel.

    Parameters mirror visualize_2d_field_magnitude, but for a single channel index.
    """
    # unpack the channel
    fields = unpack_2d_field(X[:, time_index], shape, [channel])
    fields_pred = unpack_2d_field(X_pred[:, time_index], shape, [channel])
    # plot side-by-side true, pred, and relative error
    fig, axs = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
    im0 = axs[0].imshow(fields[0].T, origin='lower'); axs[0].set_title(f'field ({tag})')
    im1 = axs[1].imshow(fields_pred[0].T, origin='lower', vmin=im0.get_clim()[0], vmax=im0.get_clim()[1]); axs[1].set_title(f'field ({tag})')
    # error calculation should avoid elements where true is zero
    valid = np.where(fields[0].T != 0)
    error = np.zeros_like(fields[0].T)
    error[valid] = np.abs(fields[0].T[valid] - fields_pred[0].T[valid]) / fields[0].T[valid]
    im2 = axs[2].imshow(error, origin='lower', vmin=0, vmax=1); axs[2].set_title('relative error')
    fig.colorbar(im0, ax=axs[0], fraction=0.046); fig.colorbar(im1, ax=axs[1], fraction=0.046); fig.colorbar(im2, ax=axs[2], fraction=0.046)
    plt.savefig(f"{tag}_{name}_at_{time_index}.png")
    plt.close()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def create_gif_from_pngs(name='2d_field', remove_pngs=True):
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
    # Optionally remove the individual PNG files
    if remove_pngs:
        for p in png_files:
            os.remove(p)

def print_error_metrics(X, X_pred, tag=""):
    """Print global relative error and per-time-step relative errors.

    - tag: optional prefix to label the evaluation block (e.g., '[test]').
    """
    global_error = print_global_error(X, X_pred, tag=tag)
    errors = []
    for k in range(X.shape[1]):
        mask = np.abs(X[:, k]) > 1e-6
        num = np.linalg.norm(X[mask, k] - X_pred[mask, k])
        den = np.linalg.norm(X[mask, k])
        relk = num / den
        print(f"  step {k:4d} relative error: {relk:.4f}")
        errors.append(relk)
    return global_error, errors

def print_global_error(X, X_pred, tag=""):
    """Compute and print global relative error ||X-X_pred|| / ||X|| with small epsilon."""
    mask = np.abs(X) > 1e-6
    global_error = np.linalg.norm(X[mask] - X_pred[mask]) / (np.linalg.norm(X[mask]))
    print(f"{tag} Global relative error: {global_error:.4f}")
    return global_error

def plot_ae_history(history, savepath="ae_loss_curve.png"):
    """Plot train vs val loss from history returned by train_autoencoder.

    Expects keys 'train_loss' and 'val_loss' in the given history dict.
    Saves a log-scaled MSE loss curve to 'savepath'.
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


def check_errors(tag, errors):
    # Check if error metrics file exists, and if yes, check whether they match
    if os.path.exists(f"{tag}_errors.txt"):
        if np.allclose(errors, np.loadtxt(f"{tag}_errors.txt"), rtol=1e-2, atol=1e-4):
            print(f"[INFO] Error metrics for {tag} match the saved file.")
        else:
            raise ValueError(f"[ERROR] Error metrics for {tag} do not match the saved file.")
    else:
        np.savetxt(f"{tag}_errors.txt", errors)