import os
import numpy as np
import matplotlib.pyplot as plt
from rom_pipeline import PodGpROM, PodSindyROM, AutoencoderSindyROM


def get_data_info():
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
    T = len(time_steps)
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
    """Plot multiple error curves vs time, slide-friendly (16:9).

    - curves: dict[label -> 1D np.ndarray of per-step errors]
    - t: time vector aligned with errors length
    - outfile: path to save the combined plot
    - group_key: optional string to prefix the title/filename
    - size: figure size in inches (width, height), default 12.8x7.2 for 1280x720 at 100-120 dpi
    - dpi: figure DPI when saving
    - legend_cols: columns for the legend layout
    """

    # Assign a color for each ROM type
    rom_colors = {
        "POD-GP": "#1f77b4",      # blue
        "POD-SINDY": "#ff7f0e",   # orange
        "AE-SINDY": "#2ca02c",    # green
    }
    # Assign line styles for each test type
    test_linestyles = {
        "recon": "solid",
        "interp n=5": "dashed",
        "interp n=10": "dashdot",
        "interp n=20": (0, (3, 1, 1, 1)),  # custom dash
        "extrap": "dotted",
    }

    def parse_label(label):
        # Extract ROM type and test type from label
        if "POD-GP" in label:
            rom = "POD-GP"
        elif "POD-SINDY" in label:
            rom = "POD-SINDY"
        elif "AE-SINDY" in label:
            rom = "AE-SINDY"
        else:
            rom = "Other"

        if "(recon)" in label:
            test = "recon"
        elif "(interp n=5)" in label:
            test = "interp n=5"
        elif "(interp n=10)" in label:
            test = "interp n=10"
        elif "(interp n=20)" in label:
            test = "interp n=20"
        elif "(extrap)" in label:
            test = "extrap"
        else:
            test = "other"
        return rom, test

    plt.figure(figsize=size)
    for label, err in curves.items():
        rom, test = parse_label(label)
        color = rom_colors.get(rom, None)
        linestyle = test_linestyles.get(test, "solid")
        if len(err) != len(t):
            m = min(len(err), len(t))
            plt.plot(t[:m], err[:m], label=label, lw=1.6, color=color, linestyle=linestyle)
        else:
            plt.plot(t, err, label=label, lw=1.6, color=color, linestyle=linestyle)
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


np.random.seed(36)
# Torch is optional; set seed if available and determine device
try:
    import torch  # type: ignore
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    device = "cpu"

file, dt, T, t = get_data_info()
channels = ["rho"]

curves_gp = run_pod_gp(file, dt, T, t, channels)
curves_ps = run_pod_sindy(file, dt, T, t, channels)
curves_ae = run_ae_sindy(file, dt, T, t, channels, device)

# Combined dict across all ROMs
combined = {**curves_gp, **curves_ps, **curves_ae}

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