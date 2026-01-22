# grainlearning/coarse_grain/plotting.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def _extent_from_grid(grid) -> Tuple[float,float,float,float]:
    return (
        grid.origin[0],
        grid.origin[0] + grid.spacing[0]*grid.shape[0],
        grid.origin[1],
        grid.origin[1] + grid.spacing[1]*grid.shape[1],
    )

def plot_scalars_2d(
    grid,
    scalars: Dict[str, np.ndarray],
    keys: Optional[List[str]] = None,
    cmap="viridis",
    save: Optional[str] = None,
    dpi: int = 150,
    show: bool = True,
):
    if keys is None:
        keys = list(scalars.keys())
    if not keys:
        raise ValueError("No scalar fields to plot.")

    if isinstance(cmap, str):
        cmaps = [cmap] * len(keys)
    else:
        if len(cmap) != len(keys):
            raise ValueError("Length of cmap list must match number of keys.")
        cmaps = cmap

    save_dir = Path(save) if save is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    extent = _extent_from_grid(grid)

    for key, cm in zip(keys, cmaps):
        if key not in scalars:
            raise KeyError(f"Key '{key}' not found in scalars dict.")
        F = scalars[key]
        if F.ndim != 2:
            raise ValueError(f"Field '{key}' is not 2D (shape {F.shape}).")

        fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
        im = ax.imshow(F.T, origin="lower", cmap=cm, extent=extent, aspect="equal")
        ax.set_title(key)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046)

        if save_dir is not None:
            fig.savefig(save_dir / f"{key}.png", dpi=dpi)
            plt.close(fig)
        elif show:
            plt.show()
        else:
            plt.close(fig)

def plot_vector_field_2d(
    grid,
    vector: Tuple[np.ndarray, np.ndarray],
    component: Optional[int] = None,
    cmap: str = "viridis",
    save: Optional[str] = None,
    dpi: int = 150,
    show: bool = True,
    basename: str = "v",
):
    vx, vy = vector
    if vx.shape != vy.shape or vx.ndim != 2:
        raise ValueError("Vector field must be 2D arrays of same shape.")

    if component is None:
        field = np.sqrt(vx**2 + vy**2)
        title = f"‖{basename}‖"; fname = f"{basename}_norm.png"
    elif component == 0:
        field = vx; title = f"{basename}_x"; fname = f"{basename}_x.png"
    elif component == 1:
        field = vy; title = f"{basename}_y"; fname = f"{basename}_y.png"
    else:
        raise ValueError("component must be None, 0, or 1.")

    extent = _extent_from_grid(grid)
    fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
    im = ax.imshow(field.T, origin="lower", cmap=cmap, extent=extent, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046)

    if save is not None:
        p = Path(save)
        if p.is_dir():
            p = p / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)

# ----------------- STRESS VISUALIZATION -----------------

def _stress_dim(stress: Tuple[Tuple[np.ndarray, ...], ...]) -> int:
    return len(stress)

def _stress_component(stress, comp: str) -> np.ndarray:
    """
    Return symmetric component array for 'xx','yy','zz','xy','xz','yz'.
    For shear, return 0.5*(sigma_ab + sigma_ba).
    """
    comp = comp.lower()
    idx_map = {"x":0, "y":1, "z":2}
    if len(comp) != 2 or comp[0] not in idx_map or comp[1] not in idx_map:
        raise ValueError(f"Invalid stress component '{comp}'. Use 'xx','xy','xz', etc.")
    a = idx_map[comp[0]]; b = idx_map[comp[1]]

    dim = _stress_dim(stress)
    if a >= dim or b >= dim:
        raise ValueError(f"Requested component '{comp}' not available for stress of dim={dim}.")

    Sab = stress[a][b]
    if Sab.ndim != 2:
        raise ValueError("Stress components must be 2D arrays for 2D plotting.")
    if a == b:
        return Sab
    # symmetrize shear
    Sba = stress[b][a]
    return 0.5*(Sab + Sba)

def _stress_trace(stress) -> np.ndarray:
    dim = _stress_dim(stress)
    tr = None
    for a in range(dim):
        Sa = stress[a][a]
        tr = Sa if tr is None else (tr + Sa)
    return tr

def _stress_mean_pressure(stress) -> np.ndarray:
    dim = _stress_dim(stress)
    tr = _stress_trace(stress)
    return -(tr / float(dim))

def _stress_deviatoric_magnitude(stress) -> np.ndarray:
    """
    J2^(1/2) = sqrt(0.5 * s:s), where s = sigma - (tr(sigma)/n) I.
    """
    dim = _stress_dim(stress)
    tr = _stress_trace(stress)
    mean = tr / float(dim)

    # build deviatoric components s_ij
    s = [[None]*dim for _ in range(dim)]
    for a in range(dim):
        for b in range(dim):
            Sij = stress[a][b]
            if a == b:
                s[a][b] = Sij - mean
            else:
                # symmetrize
                s[a][b] = 0.5*(stress[a][b] + stress[b][a])

    # s:s = sum_ij s_ij * s_ij
    acc = None
    for a in range(dim):
        for b in range(dim):
            term = s[a][b]*s[a][b]
            acc = term if acc is None else (acc + term)

    return np.sqrt(0.5 * acc)

def plot_stress_2d(
    grid,
    tensors: Dict[str, object],
    key: str,
    cmap: str = "cividis",
    save: Optional[str] = None,
    dpi: int = 150,
    show: bool = True,
    fname: Optional[str] = None,
):
    """
    Plot a stress field heatmap (2D).

    Parameters
    ----------
    grid : UniformGrid
    tensors : dict, expects tensors['stress'] = tuple-of-tuples (dim x dim) of 2D arrays
    key : str
        - Component: 'xx','yy','zz','xy','xz','yz'
        - Aggregate: 'mean' (pressure = -tr(sigma)/n), 'deviatoric' (J2^(1/2))
    """
    if "stress" not in tensors or tensors["stress"] is None:
        raise ValueError("tensors['stress'] is missing.")

    stress = tensors["stress"]
    extent = _extent_from_grid(grid)

    key_l = key.lower()
    if key_l in ("mean", "pressure"):
        F = _stress_mean_pressure(stress)
        title = "mean stress (pressure)"
        default_fname = "stress_mean.png"
    elif key_l in ("dev", "deviatoric", "j2"):
        F = _stress_deviatoric_magnitude(stress)
        title = "deviatoric magnitude (J2^1/2)"
        default_fname = "stress_deviatoric.png"
    else:
        F = _stress_component(stress, key_l)
        title = f"σ_{key_l}"
        default_fname = f"stress_{key_l}.png"

    if F.ndim != 2:
        raise ValueError(f"Stress field for '{key}' is not 2D (shape {F.shape}).")

    fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
    im = ax.imshow(F.T, origin="lower", cmap=cmap, extent=extent, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046)

    if save is not None:
        p = Path(save)
        if p.is_dir():
            p = p / (fname if fname is not None else default_fname)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)

def save_all_2d(
    grid,
    cg_out: Dict[str, dict],
    save_dir: str,
    scalar_keys: Optional[List[str]] = None,
    vector_keys: Tuple[str, ...] = ("vel", "disp"),
    vector_components: Tuple[Optional[int], ...] = (None, 0, 1),
    stress_keys: Tuple[str, ...] = ("mean", "deviatoric", "xx", "yy", "xy"),
    cmap_scalars="viridis",
    cmap_vectors="magma",
    cmap_stress="cividis",
    dpi: int = 150,
):
    """
    Batch-save:
      - Scalars: chosen keys from cg_out['scalars']
      - Vectors: chosen vector fields (norm, x, y)
      - Stress: chosen stress keys: 'mean','deviatoric','xx','yy','xy',... ('xz','yz','zz' if available)

    Filenames:
      scalars -> <key>.png
      vectors -> <vkey>_<norm|x|y>.png
      stress  -> stress_<key>.png  (or 'stress_mean.png', 'stress_deviatoric.png')
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    scalars = cg_out.get("scalars", {})
    vectors = cg_out.get("vectors", {})
    tensors = cg_out.get("tensors", {})

    # scalars
    if scalar_keys is None:
        scalar_keys = list(scalars.keys())
    if scalar_keys:
        plot_scalars_2d(grid, scalars, keys=scalar_keys, cmap=cmap_scalars, save=save_dir, dpi=dpi, show=False)

    # vectors
    for vkey in vector_keys:
        vec = vectors.get(vkey, None)
        if vec is None:
            continue
        vxvy = tuple(vec)  # (vx, vy)
        for comp in vector_components:
            out_path = save_dir / f"{vkey}_{'norm' if comp is None else ('x' if comp==0 else 'y')}.png"
            plot_vector_field_2d(
                grid, vxvy, component=comp, cmap=cmap_vectors, save=out_path, dpi=dpi, show=False, basename=vkey
            )

    # stresses
    if tensors.get("stress", None) is not None and stress_keys:
        for sk in stress_keys:
            plot_stress_2d(
                grid,
                tensors,
                key=sk,
                cmap=cmap_stress,
                save=save_dir / f"stress_{'mean' if sk.lower() in ('mean','pressure') else ('deviatoric' if sk.lower() in ('dev','deviatoric','j2') else sk.lower())}.png",
                dpi=dpi,
                show=False,
            )
