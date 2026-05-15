import glob
import importlib
import os
import re

import h5py
import numpy as np


DATA_DIR = os.path.dirname(__file__)
STEP_INTERVAL = 1000

# Output controls
SAVE_COMBINED_NPY = False
SAVE_HDF5_OUTPUT = True
MAKE_MACRO_PLOT = True

SNAPSHOT_GLOB = "column_collapse_triax_Iter0_*_*_fields.npy"
SNAPSHOT_SAMPLE_PATTERN = "column_collapse_triax_Iter0_Sample{sample_id:02d}_*_fields.npy"
SNAPSHOT_STEP_RE = re.compile(r"_(\d+)_fields\.npy$")
SAMPLE_ID_RE = re.compile(r"column_collapse_triax_Iter0_Sample(\d+)_\d+_fields\.npy$")


def _unwrap(obj):
    if isinstance(obj, np.ndarray) and obj.shape == ():
        try:
            return obj.item()
        except Exception:
            return obj
    if hasattr(obj, "item") and not isinstance(obj, dict):
        try:
            candidate = obj.item()
            if candidate is not obj:
                return candidate
        except Exception:
            pass
    return obj


def _sorted_time_keys(output_dict):
    def _sort_key(k):
        try:
            return (0, float(k))
        except Exception:
            return (1, str(k))

    return sorted(output_dict.keys(), key=_sort_key)


def _add_scalar_channels(snapshot, channels, scalar_keys):
    scalars = snapshot.get("scalars", {})
    if not isinstance(scalars, dict):
        return
    for name in scalar_keys:
        if name not in scalars:
            raise ValueError(f"Missing scalar key '{name}' in snapshot.")
        arr = np.asarray(scalars[name])
        if arr.ndim == 2:
            channels.append((f"scalars/{name}", arr))


def _add_vector_channels(snapshot, channels, vector_keys):
    vectors = snapshot.get("vectors", {})
    if not isinstance(vectors, dict):
        return

    component_labels = ["x", "y"]
    for name in vector_keys:
        if name not in vectors:
            raise ValueError(f"Missing vector key '{name}' in snapshot.")
        arr = np.asarray(vectors[name])
        if arr.ndim == 3:
            if arr.shape[0] != 2:
                raise ValueError(
                    f"Vector field '{name}' expected 2 components (x,y), got {arr.shape[0]}"
                )
            for i in range(arr.shape[0]):
                channels.append((f"vectors/{name}_{component_labels[i]}", arr[i]))


def _add_tensor_channels(snapshot, channels, tensor_keys):
    tensors = snapshot.get("tensors", {})
    if not isinstance(tensors, dict):
        return

    component_labels = ["x", "y"]
    for name in tensor_keys:
        if name not in tensors:
            raise ValueError(f"Missing tensor key '{name}' in snapshot.")
        arr = np.asarray(tensors[name])
        if arr.ndim == 4:
            if arr.shape[0] != 2 or arr.shape[1] != 2:
                raise ValueError(
                    f"Tensor field '{name}' expected shape (2,2,x,y), got {arr.shape}"
                )
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    channels.append((f"tensors/{name}_{component_labels[i]}{component_labels[j]}", arr[i, j]))
        elif arr.ndim == 3:
            if arr.shape[0] != 4:
                raise ValueError(
                    f"Tensor field '{name}' componentized form expected 4 components, got {arr.shape[0]}"
                )
            component_order = ["xx", "xy", "yx", "yy"]
            for i in range(arr.shape[0]):
                channels.append((f"tensors/{name}_{component_order[i]}", arr[i]))


def _extract_channels(snapshot, scalar_keys, vector_keys, tensor_keys):
    channels = []
    _add_scalar_channels(snapshot, channels, scalar_keys)
    _add_vector_channels(snapshot, channels, vector_keys)
    _add_tensor_channels(snapshot, channels, tensor_keys)
    return channels


def _build_4d_array(sim_data):
    time_keys = _sorted_time_keys(sim_data)
    if not time_keys:
        raise ValueError("Empty simulation dictionary.")

    first_snapshot = _unwrap(sim_data[time_keys[0]])
    if not isinstance(first_snapshot, dict):
        raise ValueError("Snapshot format is not a dictionary.")

    scalar_dict = first_snapshot.get("scalars", {})
    vector_dict = first_snapshot.get("vectors", {})
    tensor_dict = first_snapshot.get("tensors", {})

    scalar_keys = list(scalar_dict.keys()) if isinstance(scalar_dict, dict) else []
    vector_keys = list(vector_dict.keys()) if isinstance(vector_dict, dict) else []
    tensor_keys = list(tensor_dict.keys()) if isinstance(tensor_dict, dict) else []

    channels0 = _extract_channels(first_snapshot, scalar_keys, vector_keys, tensor_keys)
    if not channels0:
        raise ValueError("No scalar/vector/tensor channels found in snapshot.")

    channel_names = [name for name, _ in channels0]
    ref_shape = channels0[0][1].shape

    for name, arr in channels0:
        if arr.shape != ref_shape:
            raise ValueError(f"Inconsistent channel shape in first snapshot: {name} -> {arr.shape}")

    t_count = len(time_keys)
    c_count = len(channel_names)
    x_size, y_size = ref_shape
    data = np.empty((t_count, c_count, x_size, y_size), dtype=np.float32)

    for t_idx, tk in enumerate(time_keys):
        snapshot = _unwrap(sim_data[tk])
        if not isinstance(snapshot, dict):
            raise ValueError(f"Snapshot at time key {tk!r} is not a dictionary.")

        channels = _extract_channels(snapshot, scalar_keys, vector_keys, tensor_keys)
        names = [n for n, _ in channels]
        if names != channel_names:
            raise ValueError(
                "Channel set changed across time steps. "
                f"Expected {channel_names}, got {names} at key {tk!r}."
            )

        for c_idx, (_, arr) in enumerate(channels):
            if arr.shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch at time key {tk!r}, channel {channel_names[c_idx]}: "
                    f"expected {ref_shape}, got {arr.shape}."
                )
            data[t_idx, c_idx] = arr.astype(np.float32, copy=False)

    time_vals = []
    for i, k in enumerate(time_keys):
        try:
            time_vals.append(float(k))
        except Exception:
            time_vals.append(float(i))
    time_array = np.array(time_vals, dtype=np.float64)
    metadata = {
        "scalar_keys": scalar_keys,
        "vector_keys": vector_keys,
        "tensor_keys": tensor_keys,
    }
    return data, channel_names, time_array, metadata

def _discover_sample_ids(data_dir):
    snapshot_files = glob.glob(os.path.join(data_dir, SNAPSHOT_GLOB))
    sample_ids = []
    for f in snapshot_files:
        m = SAMPLE_ID_RE.search(os.path.basename(f))
        if m:
            sample_ids.append(int(m.group(1)))
    sample_ids = sorted(set(sample_ids))
    print(f"Found {len(snapshot_files)} snapshot files corresponding to {len(sample_ids)} samples.")
    return sample_ids


def combine_data_per_sample(sample_id, step_interval=STEP_INTERVAL):
    """Combine per-step snapshot files for one sample into an indexed dict."""
    combined_data = {}
    pattern = os.path.join(DATA_DIR, SNAPSHOT_SAMPLE_PATTERN.format(sample_id=sample_id))
    source_files = glob.glob(pattern)
    source_files = sorted(
        source_files,
        key=lambda x: int(SNAPSHOT_STEP_RE.search(os.path.basename(x)).group(1)),
    )
    if not source_files:
        raise FileNotFoundError(f"No snapshots found for sample {sample_id:02d} with pattern: {pattern}")

    time_steps = [int(SNAPSHOT_STEP_RE.search(os.path.basename(f)).group(1)) for f in source_files]
    if not all(t2 - t1 == step_interval for t1, t2 in zip(time_steps, time_steps[1:])):
        raise ValueError(
            f"Time steps for sample {sample_id:02d} are not increasing by {step_interval}: {time_steps}"
        )
    expected = list(range(time_steps[0], time_steps[0] + step_interval * len(time_steps), step_interval))
    if time_steps != expected:
        raise ValueError(f"Time steps for sample {sample_id:02d} are not continuous: {time_steps}")

    for i, fname in enumerate(source_files):
        data = np.load(fname, allow_pickle=True)
        combined_data[i] = data
    if SAVE_COMBINED_NPY:
        np.save(os.path.join(DATA_DIR, f"column_collapse_{sample_id:02d}_CG_fields.npy"), combined_data)
    return combined_data, source_files


def save_combined_hdf5(sample_id, combined_data, source_files):
    data, channel_names, time_array, metadata = _build_4d_array(combined_data)
    out_path = os.path.join(DATA_DIR, f"column_collapse_{sample_id:02d}_CG_fields.hdf5")
    with h5py.File(out_path, "w") as h5f:
        data_ds = h5f.create_dataset("data", data=data, compression="gzip", compression_opts=4)
        h5f.create_dataset("time", data=time_array)
        h5f.create_dataset(
            "source_files",
            data=np.array([os.path.basename(f) for f in source_files], dtype=h5py.string_dtype("utf-8")),
        )
        h5f.create_dataset("channel_names", data=np.array(channel_names, dtype=h5py.string_dtype("utf-8")))

        str_dtype = h5py.string_dtype("utf-8")
        data_ds.attrs["scalar_keys"] = np.array(metadata["scalar_keys"], dtype=str_dtype)
        data_ds.attrs["vector_keys"] = np.array(metadata["vector_keys"], dtype=str_dtype)
        data_ds.attrs["tensor_keys"] = np.array(metadata["tensor_keys"], dtype=str_dtype)

    # verification: compare directly against raw snapshot files on disk
    with h5py.File(out_path, "r") as h5f:
        got_data = h5f["data"]
        got_time = h5f["time"][...]
        got_names = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in h5f["channel_names"][...]]

        # expected channel names from first raw snapshot
        first_raw = _unwrap(np.load(source_files[0], allow_pickle=True))
        if not isinstance(first_raw, dict):
            raise ValueError(f"Invalid snapshot format in {source_files[0]}")

        scalar_dict = first_raw.get("scalars", {})
        vector_dict = first_raw.get("vectors", {})
        tensor_dict = first_raw.get("tensors", {})
        scalar_keys = list(scalar_dict.keys()) if isinstance(scalar_dict, dict) else []
        vector_keys = list(vector_dict.keys()) if isinstance(vector_dict, dict) else []
        tensor_keys = list(tensor_dict.keys()) if isinstance(tensor_dict, dict) else []

        first_channels = _extract_channels(first_raw, scalar_keys, vector_keys, tensor_keys)
        exp_channel_names = [n for n, _ in first_channels]
        exp_shape = first_channels[0][1].shape
        exp_data_shape = (len(source_files), len(exp_channel_names), exp_shape[0], exp_shape[1])

        if got_data.shape != exp_data_shape:
            raise ValueError(
                f"Shape mismatch for sample {sample_id:02d}: h5 {got_data.shape} vs expected {exp_data_shape}"
            )

        exp_time_array = np.arange(len(source_files), dtype=np.float64)
        if not np.array_equal(got_time, exp_time_array):
            raise ValueError(f"Time mismatch for sample {sample_id:02d}")
        if got_names != exp_channel_names:
            raise ValueError(f"Channel name mismatch for sample {sample_id:02d}")

        max_abs_diff = 0.0
        for t_idx, fname in enumerate(source_files):
            raw_snapshot = _unwrap(np.load(fname, allow_pickle=True))
            if not isinstance(raw_snapshot, dict):
                raise ValueError(f"Invalid snapshot format in {fname}")

            channels = _extract_channels(raw_snapshot, scalar_keys, vector_keys, tensor_keys)
            names = [n for n, _ in channels]
            if names != exp_channel_names:
                raise ValueError(
                    f"Channel mismatch at t={t_idx} for sample {sample_id:02d}: expected {exp_channel_names}, got {names}"
                )

            exp_snapshot = np.stack([arr.astype(np.float32, copy=False) for _, arr in channels], axis=0)
            diff = float(np.max(np.abs(got_data[t_idx] - exp_snapshot)))
            if diff > max_abs_diff:
                max_abs_diff = diff

    if max_abs_diff != 0.0:
        raise ValueError(f"Value mismatch for sample {sample_id:02d}, max_abs_diff={max_abs_diff}")

    print(f"Saved and verified HDF5: {out_path} (shape={data.shape}, max_abs_diff=0.0)")

def plot_macro_measures(data_dir):
    txt_files = sorted(glob.glob(os.path.join(data_dir, "column_collapse_triax_Iter0_*.txt")))
    if not txt_files:
        print("No macro trajectory text files found. Skipping plot.")
        return

    plt = importlib.import_module("matplotlib.pyplot")
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    for i, fname in enumerate(txt_files):
        data = np.loadtxt(fname, skiprows=1)
        label = f"SampleID_{i}"
        axs[0, 0].plot(data[:, -1], data[:, 0], label=label)
        axs[0, 1].plot(data[:, -1], data[:, 1], label=label)
        axs[1, 0].plot(data[:, -1], data[:, 2], label=label)
        axs[1, 1].plot(data[:, -1], data[:, 3], label=label)

    axs[0, 0].set_xlabel("time"); axs[0, 0].set_ylabel("com_x (m)")
    axs[0, 1].set_xlabel("time"); axs[0, 1].set_ylabel("com_y (m)")
    axs[1, 0].set_xlabel("time"); axs[1, 0].set_ylabel("height (m)")
    axs[1, 1].set_xlabel("time"); axs[1, 1].set_ylabel("runout (m)")
    plt.savefig(os.path.join(data_dir, "macro_measures.png"))
    plt.close(fig)


def main():
    sample_ids = _discover_sample_ids(DATA_DIR)
    for sample_id in sample_ids:
        combined_data, source_files = combine_data_per_sample(sample_id)
        if SAVE_HDF5_OUTPUT:
            save_combined_hdf5(sample_id, combined_data, source_files)
        print(f"Combined data for sample {sample_id} saved.")

    if MAKE_MACRO_PLOT:
        plot_macro_measures(DATA_DIR)


if __name__ == "__main__":
    main()