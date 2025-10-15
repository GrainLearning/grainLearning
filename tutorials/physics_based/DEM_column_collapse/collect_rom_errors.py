import os
import numpy as np
import matplotlib.pyplot as plt
from rom_pipeline import PodGpROM, PodSindyROM, AutoencoderSindyROM, ParametricPodSindyROM
import matplotlib.tri as mtri
import matplotlib as mpl


def get_data_info(t_max=None):
    data_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "Yade",
            "column_collapse_sim_data",
            "column_collapse_15_CG_fields.npy",
        )
    )
    output = np.load(data_file, allow_pickle=True).item()
    time_steps = list(output.keys())
    dt = 1.97e-5 * (time_steps[1] - time_steps[0])
    T_full = len(time_steps)
    T = min(T_full, int(t_max)) if t_max is not None else T_full
    t = np.arange(T) * dt
    return data_file, dt, T, t


def run_pod_gp(file, dt, T, t, channels):
    curves = {}
    # Reconstruction on training data
    rom = PodGpROM(normalization=True, energy=0.99, num_modes=10, tag="POD-GP")
    rom.fit(file, channels=channels, dt=dt)
    rom.evaluate(file, t, create_visual=False)
    curves["POD-GP (recon)"] = np.array(rom.errors)

    # Interpolation for n in [5, 10, 20]
    n_values = [5, 10, 20]
    # Preload full data once for slicing
    full_channels = rom.channel_list
    for n in n_values:
        tag = "POD-GP_interpolate" if n == 5 else f"POD-GP_interpolate_n{n}"
        X_train = [ch[::n, :, :] for ch in full_channels]
        rom.tag = tag
        rom.fit(X_train, channels=channels, dt=n * dt)
        rom.evaluate(file, t, create_visual=False)
        curves[f"POD-GP (interp n={n})"] = np.array(rom.errors)

    # Extrapolation (train on first half, evaluate on full)
    tag = "POD-GP_extrapolate"
    mid = T // 2
    X_train = [ch[:mid, :, :] for ch in full_channels]
    rom.tag = tag
    rom.fit(X_train, channels=channels, dt=dt)
    rom.evaluate(file, t, create_visual=False)
    curves["POD-GP (extrap)"] = np.array(rom.errors)
    return curves


def run_pod_sindy(file, dt, T, t, channels):
    curves = {}
    # Reconstruction on training data
    rom = PodSindyROM(
        normalization=True,
        energy=0.99,
        num_modes=3,
        poly_degree=2,
        thresh=0.1,
        diff="smoothed",
        tag="POD-SINDY",
    )
    rom.fit(data_or_file=file, channels=channels, dt=dt)
    # Match original baseline evaluation using preloaded channel_list
    rom.evaluate(rom.channel_list, t, create_visual=False)
    curves["POD-SINDY (recon)"] = np.array(rom.errors)

    # Interpolation for n in [5, 10, 20]
    n_values = [5, 10, 20]
    full_channels = rom.channel_list
    for n in n_values:
        tag = "POD-SINDY_interpolate" if n == 5 else f"POD-SINDY_interpolate_n{n}"
        X_train = [ch[::n, :, :] for ch in full_channels]
        rom.tag = tag
        rom.fit(X_train, channels=channels, dt=n * dt)
        rom.evaluate(file, t, create_visual=False)
        curves[f"POD-SINDY (interp n={n})"] = np.array(rom.errors)

    # Extrapolation (train on first half, evaluate on full)
    tag = "POD-SINDY_extrapolate"
    mid = T // 2
    X_train = [ch[:mid, :, :] for ch in full_channels]
    rom.tag = tag
    rom.fit(X_train, channels=channels, dt=dt)
    rom.evaluate(file, t, create_visual=False)
    curves["POD-SINDY (extrap)"] = np.array(rom.errors)
    return curves


def run_ae_sindy(file, dt, T, t, channels, device: str):
    curves = {}
    # Reconstruction on training data (use CPU by default for portability)
    rom = AutoencoderSindyROM(
        normalization=True,
        latent_dim=3,
        poly_degree=1,
        thresh=0.1,
        diff="smoothed",
        epochs=5000,
        batch_size=64,
        lr=1e-4,
        device=device,
        val_split=0.2,
        patience=200,
        print_every=50,
        tag="AE-SINDY",
    )
    rom.fit(data_or_file=file, channels=channels, dt=dt)
    # Match original baseline evaluation using preloaded channel_list
    rom.evaluate(rom.channel_list, t, create_visual=False)
    curves["AE-SINDY (recon)"] = np.array(rom.errors)

    # Interpolation for n in [5, 10, 20]
    n_values = [5, 10, 20]
    full_channels = rom.channel_list
    for n in n_values:
        tag = f"AE-SINDY_interpolate_n{n}"
        X_train = [ch[::n, :, :] for ch in full_channels]
        rom.tag = tag
        rom.fit(X_train, channels=channels, dt=n * dt)
        rom.evaluate(file, t, create_visual=False)
        curves[f"AE-SINDY (interp n={n})"] = np.array(rom.errors)

    # Extrapolation (train on first half, evaluate on full)
    tag = "AE-SINDY_extrapolate"
    mid = T // 2
    X_train = [ch[:mid, :, :] for ch in full_channels]
    rom.tag = tag
    rom.fit(X_train, channels=channels, dt=dt)
    rom.evaluate(file, t, create_visual=False)
    curves["AE-SINDY (extrap)"] = np.array(rom.errors)
    return curves


def plot_errors(curves, t, outfile="rom_errors_over_time.png", group_key=None,
                size=(9.6, 5.4), dpi=150, legend_cols=2, save_pdf=False):
    """
    Plot multiple error curves vs time, slide-friendly (16:9), with consistent colors per ROM.
    Interpolation lines for each ROM use the same base color, with decreasing intensity as n increases.
    Extrapolation uses the same color as the first interpolation for each ROM.
    """

    # Define ROM base colors
    rom_colors = {
        "POD-GP": "#1f77b4",      # blue
        "POD-SINDY": "#2ca02c",   # green
        "AE-SINDY": "#d62728",    # red
    }
    # For each ROM, define interpolation n values (must match code above)
    interp_ns = [5, 10, 20]

    # Helper to extract ROM name from label
    def get_rom_name(label):
        for rom in rom_colors:
            if label.startswith(rom):
                return rom
        # fallback: try to match by substring
        for rom in rom_colors:
            if rom in label:
                return rom
        return None

    # For each ROM, collect all relevant curve keys
    rom_curve_keys = {rom: [] for rom in rom_colors}
    for k in curves:
        rom = get_rom_name(k)
        if rom:
            rom_curve_keys[rom].append(k)

    # For each ROM, assign colors for interpolation/extrapolation
    color_map = {}
    for rom, base_color in rom_colors.items():
        # Use a colormap to generate lighter shades for higher n
        cmap = mpl.colormaps.get_cmap("Blues" if rom == "POD-GP" else
                                      "Greens" if rom == "POD-SINDY" else
                                      "Reds")
        # Assign colors for interpolation n=5,10,20 (darker to lighter)
        interp_colors = [cmap(0.7), cmap(0.5), cmap(0.3)]
        # Map keys to colors
        for i, n in enumerate(interp_ns):
            for k in rom_curve_keys[rom]:
                if f"(interp n={n})" in k or f"_interpolate_n{n}" in k or (n == 5 and "_interpolate" in k and "n" not in k):
                    color_map[k] = interp_colors[i]
        # Extrapolation: use color of n=5
        for k in rom_curve_keys[rom]:
            if "(extrap)" in k or "_extrapolate" in k:
                color_map[k] = interp_colors[0]
        # Recon: use base color
        for k in rom_curve_keys[rom]:
            if "(recon)" in k:
                color_map[k] = base_color

    plt.figure(figsize=size)
    for label, err in curves.items():
        if len(err) != len(t):
            m = min(len(err), len(t))
            x = t[:m]
            y = err[:m]
        else:
            x = t
            y = err
        color = color_map.get(label, None)
        plt.plot(x, y, label=label, lw=1.6, color=color)
    plt.xlabel("time")
    plt.ylabel("relative error")
    if group_key:
        plt.title(f"ROM errors over time - {group_key}")
    else:
        plt.title("ROM errors over time")
    plt.yscale("log")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(fontsize=8, ncol=legend_cols, frameon=False)
    plt.tight_layout()
    plt.margins(x=0)
    plt.savefig(outfile, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        root, _ = os.path.splitext(outfile)
        plt.savefig(root + ".pdf", bbox_inches="tight")
    plt.close()


def load_parametric_dataset(t_max=None):
    """Load parametric files and corresponding parameter vectors (3 params)."""
    data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "Yade", "column_collapse_sim_data"))
    file_list = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith("_CG_fields.npy") and f.startswith("column_collapse_")
    ])
    # Load parameter values (skip header row, use columns 3,4,5 as in test)
    param_file = os.path.join(data_dir, "collapse_Iter0_Samples.txt")
    u_all = np.loadtxt(param_file, skiprows=1, usecols=range(3, 6))

    # Align lengths (trim to min in case of mismatch)
    N = min(len(file_list), len(u_all))
    file_list = file_list[:N]
    u_all = u_all[:N]

    # Extract dt and T from first file
    out0 = np.load(file_list[0], allow_pickle=True).item()
    time_steps = list(out0.keys())
    dt = 1.97e-5 * (time_steps[1] - time_steps[0])
    T_full = len(time_steps)
    T = min(T_full, int(t_max)) if t_max is not None else T_full
    t = np.arange(T) * dt
    return file_list, u_all, dt, T, t


def evaluate_parametric_scenario(file_list, u_all, t, dt, train_count,
                                 channels=("rho",), t_max=None,
                                 create_visual: bool = False, every: int = 10):
    """Fit ParametricPodSindyROM with train_count runs and evaluate error on all runs.

    Parameters:
    - file_list: list of npy files for all runs
    - u_all: array of shape (N_runs, 3) with parameter triplets
    - t: time vector for evaluation
    - dt: time step used for training the ROM
    - train_count: number of runs to use for training (prefix subset)
    - channels: tuple/list of channels to use (default: ("rho",))
    """
    train_count = min(train_count, len(file_list))
    file_train = file_list[:train_count]
    u_train = u_all[:train_count]

    # Fit model
    rom = ParametricPodSindyROM(normalization=True, energy=0.99, num_modes=4,
                                poly_deg_state=1, poly_deg_param=1, thresh=0.1,
                                diff="smoothed", tag=f"POD-SINDy_Param_{train_count}")
    rom.fit(file_list=file_train, u_list=list(u_train), channels=list(channels), dt=dt, t_max=t_max)

    # Evaluate across all runs
    errors = rom.evaluate_parametric(
        file_list=file_list,
        u_list_new=list(u_all),
        t=t,
        create_visual=create_visual,
        every=every,
    )
    return np.array(errors), np.arange(train_count)


def plot_parametric_error_contours(u_all, errors, train_indices, outfile_prefix,
                                   titlesuffix="", size=(11.5, 4.5), dpi=150,
                                   param_names=("k_r", "eta", "mu")):
    """Plot 1x3 panels of error tricontours over parameter pairs with training points marked."""
    # Ensure arrays
    u_all = np.asarray(u_all)
    errors = np.asarray(errors)

    # Parameter pairs (0-1, 0-2, 1-2)
    pairs = [(0, 1), (0, 2), (1, 2)]

    fig, axes = plt.subplots(1, 3, figsize=size, constrained_layout=True)

    # Normalize color scale across panels for consistency
    vmin, vmax = float(np.min(errors)), float(np.max(errors))

    for ax, (i, j) in zip(axes, pairs):
        x = u_all[:, i]
        y = u_all[:, j]
        triang = mtri.Triangulation(x, y)
        cs = ax.tricontourf(
            triang,
            errors,
            levels=12,
            cmap="Greys",
            vmin=vmin,
            vmax=vmax,
            alpha=0.6,  # make contours semi-transparent for white backgrounds
        )

        # Overlay training points
        ax.scatter(
            u_all[train_indices, i],
            u_all[train_indices, j],
            facecolor="none",
            edgecolor="k",
            s=40,
            linewidths=1.2,
            label="train",
        )
        ax.set_xlabel(param_names[i])
        ax.set_ylabel(param_names[j])
        ax.grid(True, ls=":", alpha=0.4)

    # Shared colorbar
    cbar = fig.colorbar(cs, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Global relative error")

    if titlesuffix:
        fig.suptitle(f"Parametric error contours - {titlesuffix}")

    # Save
    png = f"{outfile_prefix}.png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def time_dependent_rom():
    file, dt, T, t = get_data_info()
    channels = ["rho"]

    curves_gp = run_pod_gp(file, dt, T, t, channels)
    curves_ps = run_pod_sindy(file, dt, T, t, channels)
    curves_ae = run_ae_sindy(file, dt, T, t, channels, device)

    # Combined dict across all ROMs
    combined = {**curves_gp, **curves_ps, **curves_ae}

    # Plot reconstruction-only (across all ROMs)
    recon_all = {k: v for k, v in combined.items() if "(recon)" in k}
    if recon_all:
        plot_errors(
            recon_all,
            t,
            outfile="rom_errors_reconstruction.png",
            group_key="Reconstruction (all ROMs)",
            size=(5.5, 4)
        )

    # Plot interpolation-only (across all ROMs)
    interp_all = {k: v for k, v in combined.items() if "(interp" in k}
    if interp_all:
        plot_errors(
            interp_all,
            t,
            outfile="rom_errors_interpolation.png",
            group_key="Interpolation (all ROMs)",
            size =(5.5, 4)
        )

    # Plot extrapolation-only (across all ROMs)
    extrap_all = {k: v for k, v in combined.items() if "(extrap)" in k}
    if extrap_all:
        plot_errors(
            extrap_all,
            t,
            outfile="rom_errors_extrapolation.png",
            group_key="Extrapolation (all ROMs)",
            size =(5.5, 4)
        )

    # Print the average errors per ROM type and test type
    for rom in ["POD-GP", "POD-SINDY", "AE-SINDY"]:
        for test_type, key_substr in [
            ("recon", "(recon)"),
            ("interp n=5", "(interp n=5)"),
            ("interp n=10", "(interp n=10)"),
            ("interp n=20", "(interp n=20)"),
            ("extrap", "(extrap)")
        ]:
            relevant_keys = [k for k in combined if k.startswith(rom) and key_substr in k]
            if relevant_keys:
                all_errors = np.concatenate([combined[k] for k in relevant_keys])
                avg_error = np.mean(all_errors[1:])  # skip first value
                print(f"{rom} average {test_type} error over all runs: {avg_error:.4f}")


def parameter_dependent_rom():
    # Set this to an integer (e.g., 200) to truncate time series, or None for full length
    t_max = 400

    # Run parametric scenarios and plot
    file_list_param, u_all, dt_p, T_p, t_p = load_parametric_dataset(t_max=t_max)
    total = len(file_list_param)
    # Desired training sizes (32, 16, 8) min with available
    desired = [32, 16, 8]
    train_sizes = [k for k in desired if k <= total]
    if not train_sizes:
        # fallback fractions if dataset smaller
        train_sizes = sorted(list(set([max(1, total//1), max(1, total//2), max(1, total//4)])), reverse=True)

    for ntrain in train_sizes:
        errors_all, train_idx = evaluate_parametric_scenario(
            file_list_param,
            u_all,
            t_p,
            dt_p,
            ntrain,
            channels=("rho","disp_x","disp_y"),
            t_max=t_max,
            create_visual=False
        )
        title = f"train={ntrain}/{total} runs"
        outpref = f"param_errors_train_{ntrain}"
        plot_parametric_error_contours(u_all, errors_all, train_idx, outpref, titlesuffix=title,
                                        size=(11.5, 4.5), dpi=150)


np.random.seed(36)
try:
    import torch  # type: ignore
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    device = "cpu"

time_dependent_rom()
parameter_dependent_rom()