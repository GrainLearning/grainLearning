# grainlearning/coarse_grain/CG.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any

Array = np.ndarray

@dataclass(frozen=True)
class UniformGrid:
    origin: Tuple[float, ...]   # world units (x0, y0[, z0])
    spacing: Tuple[float, ...]  # world units (dx, dy[, dz])
    shape:   Tuple[int,  ...]   # cells (nx, ny[, nz])

    def __post_init__(self):
        dim = len(self.shape)
        if dim not in (2, 3):
            raise ValueError("Grid must be 2D or 3D.")
        if len(self.origin) != dim or len(self.spacing) != dim:
            raise ValueError("origin/spacing dimension mismatch.")
        object.__setattr__(self, "dim", dim)
        # total domain lengths (for periodic minimal image)
        L = tuple(self.spacing[a] * self.shape[a] for a in range(dim))
        object.__setattr__(self, "domain_lengths", L)
        # cell volume
        vol = 1.0
        for d in self.spacing:
            vol *= d
        object.__setattr__(self, "cell_vol", vol)

def _as_tuple(x, dim):
    if np.isscalar(x): return (float(x),) * dim
    x = tuple(float(v) for v in x)
    if len(x) != dim: raise ValueError(f"Expected length {dim}, got {len(x)}")
    return x

def _periodic_delta(d: Array, L: float) -> Array:
    """Minimal-image shift for distance d on a periodic axis with length L."""
    return (d + 0.5 * L) % L - 0.5 * L

def _gauss_norm_separable(w: Sequence[float]) -> float:
    """Normalization for separable Gaussian ∏_k [1/(√(2π) w_k)] in R^d."""
    w = np.array(w, dtype=float)
    return 1.0 / ((np.sqrt(2.0*np.pi) ** w.size) * np.prod(w))

def _leggauss_01(n: int):
    """Gauss–Legendre nodes/weights on [0,1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    s = 0.5 * (x + 1.0)
    ws = 0.5 * w
    return s, ws

def coarse_grain(
    grid: UniformGrid,
    *,
    # particles (current time)
    ids: Array,                   # (N,) global IDs for mapping contacts -> local indices
    pos: Array,                   # (N, dim)
    mass: Array,                  # (N,)
    vel: Optional[Array] = None,  # (N, dim)
    pos_ref: Optional[Array] = None,  # (N, dim) for displacement = pos - pos_ref
    radii: Optional[Array] = None,    # (N,) for volume fraction
    # contacts (current time) for contact stress
    contacts_i: Optional[Array] = None,        # (M,) global IDs i
    contacts_j: Optional[Array] = None,        # (M,) global IDs j
    contact_forces: Optional[Array] = None,    # (M, dim) force on i from j
    # kernel / options
    w_len: float | Sequence[float] = 1.5,      # Gaussian width(s) in world units (per axis)
    cutoff_c: float = 3.0,                     # finite support radius r <= c * w
    periodic: bool | Sequence[bool] = False,   # True/False or per-axis tuple
    stress_quad: int = 3,                      # GL points along branch for contact stress
    compute_scalars: bool = True,
    compute_vectors: bool = True,
    compute_stress: bool = True,
    compute_stiffness: bool = False,  # (not implemented)
) -> Dict[str, Any]:
    """
    Weinhart/Goldhirsch coarse-graining for DEM -> Eulerian fields.

    Returns:
      {
        'scalars': {'phi', 'rho', 'occ'},
        'vectors': {'disp': (d arrays) or None, 'vel': (d arrays) or None},
        'tensors': {'stress': (d x d tuple of arrays)}  # Cauchy stress (kinetic + contact)
        'params' : {w_len, cutoff_c, periodic}
      }
    """
    dim = grid.dim
    N = int(pos.shape[0]); assert pos.shape[1] == dim
    mass = np.asarray(mass).reshape(-1); assert mass.size == N
    ids = np.asarray(ids).reshape(-1);   assert ids.size == N
    if vel is not None:
        vel = np.asarray(vel); assert vel.shape == (N, dim)
    if pos_ref is not None:
        pos_ref = np.asarray(pos_ref); assert pos_ref.shape == (N, dim)
    if radii is not None:
        radii = np.asarray(radii).reshape(-1); assert radii.size == N

    if periodic is True: periodic = (True,) * dim
    elif periodic is False: periodic = (False,) * dim
    else:
        periodic = tuple(bool(b) for b in periodic); assert len(periodic) == dim

    # kernel parameters
    w = np.array(_as_tuple(w_len, dim), dtype=float)  # world units
    if np.any(w <= 0): raise ValueError("w_len must be > 0")
    cutoff2 = (cutoff_c ** 2)
    Cw = _gauss_norm_separable(w)                     # separable Gaussian normalization
    L = grid.domain_lengths

    # allocate outputs
    rho = np.zeros(grid.shape) if compute_scalars else None
    phi = np.zeros(grid.shape) if (compute_scalars and (radii is not None)) else None
    occ = np.zeros(grid.shape)  # occupancy / kernel mass (always built)

    disp = tuple(np.zeros(grid.shape) for _ in range(dim)) if (compute_vectors and pos_ref is not None) else None
    mom  = [np.zeros(grid.shape) for _ in range(dim)] if (compute_vectors and vel is not None) else None

    S = [[np.zeros(grid.shape) for _ in range(dim)] for __ in range(dim)] if compute_stress else None

    # -------- per-particle accumulation (mass, momentum, phi, occ, disp) --------
    for i in range(N):
        # local per-axis index ranges
        idx_ranges = []
        per_axis_centers = []
        for a in range(dim):
            xc_idx = (pos[i, a] - grid.origin[a]) / grid.spacing[a]
            support_cells = cutoff_c * (w[a] / grid.spacing[a])
            lo = max(int(np.floor(xc_idx - support_cells)), 0)
            hi = min(int(np.ceil (xc_idx + support_cells)), grid.shape[a] - 1)
            idx = np.arange(lo, hi + 1, dtype=int)
            idx_ranges.append(idx)
            per_axis_centers.append(grid.origin[a] + (idx + 0.5) * grid.spacing[a])

        if dim == 2:
            ix, iy = idx_ranges
            x = per_axis_centers[0]; y = per_axis_centers[1]
            X, Y = np.meshgrid(x, y, indexing='ij')
            dx = X - pos[i, 0]; dy = Y - pos[i, 1]
            if periodic[0]: dx = _periodic_delta(dx, L[0])
            if periodic[1]: dy = _periodic_delta(dy, L[1])
            r2 = (dx*dx)/(w[0]**2) + (dy*dy)/(w[1]**2)
            W = np.where(r2 <= cutoff2, Cw * np.exp(-0.5 * r2), 0.0)
            sl = np.ix_(ix, iy)

            if compute_scalars:
                rho[sl] += mass[i] * W
            occ[sl] += W
            if (compute_scalars and radii is not None):
                vol_i = np.pi * radii[i]**2
                phi[sl] += vol_i * W / grid.cell_vol

            if (compute_vectors and vel is not None):
                mom[0][sl] += mass[i] * vel[i, 0] * W
                mom[1][sl] += mass[i] * vel[i, 1] * W
            if (compute_vectors and pos_ref is not None):
                disp[0][sl] += (pos[i, 0] - pos_ref[i, 0]) * W
                disp[1][sl] += (pos[i, 1] - pos_ref[i, 1]) * W

        else:
            ix, iy, iz = idx_ranges
            x = per_axis_centers[0]; y = per_axis_centers[1]; z = per_axis_centers[2]
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            dx = X - pos[i, 0]; dy = Y - pos[i, 1]; dz = Z - pos[i, 2]
            if periodic[0]: dx = _periodic_delta(dx, L[0])
            if periodic[1]: dy = _periodic_delta(dy, L[1])
            if periodic[2]: dz = _periodic_delta(dz, L[2])
            r2 = (dx*dx)/(w[0]**2) + (dy*dy)/(w[1]**2) + (dz*dz)/(w[2]**2)
            W = np.where(r2 <= cutoff2, Cw * np.exp(-0.5 * r2), 0.0)
            sl = np.ix_(ix, iy, iz)

            if compute_scalars:
                rho[sl] += mass[i] * W
            occ[sl] += W
            if (compute_scalars and radii is not None):
                vol_i = (4.0/3.0) * np.pi * radii[i]**3
                phi[sl] += vol_i * W / grid.cell_vol

            if (compute_vectors and vel is not None):
                mom[0][sl] += mass[i] * vel[i, 0] * W
                mom[1][sl] += mass[i] * vel[i, 1] * W
                mom[2][sl] += mass[i] * vel[i, 2] * W
            if (compute_vectors and pos_ref is not None):
                disp[0][sl] += (pos[i, 0] - pos_ref[i, 0]) * W
                disp[1][sl] += (pos[i, 1] - pos_ref[i, 1]) * W
                disp[2][sl] += (pos[i, 2] - pos_ref[i, 2]) * W

    vel_field = None
    if (compute_vectors and vel is not None and compute_scalars):
        vel_field = tuple(np.divide(mom[a], rho, out=np.zeros_like(mom[a]), where=(rho > 0))
                          for a in range(dim))

    # -------- stress = kinetic + contact --------
    stress_field = None
    if compute_stress:
        # Kinetic: -Σ m_i (v_i - u) ⊗ (v_i - u) W
        if vel is not None and compute_scalars:
            u = vel_field if vel_field is not None else tuple(np.zeros_like(rho) for _ in range(dim))
            for i in range(N):
                idx_ranges = []
                per_axis_centers = []
                for a in range(dim):
                    xc_idx = (pos[i, a] - grid.origin[a]) / grid.spacing[a]
                    support_cells = cutoff_c * (w[a] / grid.spacing[a])
                    lo = max(int(np.floor(xc_idx - support_cells)), 0)
                    hi = min(int(np.ceil (xc_idx + support_cells)), grid.shape[a] - 1)
                    idx = np.arange(lo, hi + 1, dtype=int)
                    idx_ranges.append(idx)
                    per_axis_centers.append(grid.origin[a] + (idx + 0.5) * grid.spacing[a])

                if dim == 2:
                    ix, iy = idx_ranges
                    x = per_axis_centers[0]; y = per_axis_centers[1]
                    X, Y = np.meshgrid(x, y, indexing='ij')
                    dx = X - pos[i, 0]; dy = Y - pos[i, 1]
                    if periodic[0]: dx = _periodic_delta(dx, L[0])
                    if periodic[1]: dy = _periodic_delta(dy, L[1])
                    r2 = (dx*dx)/(w[0]**2) + (dy*dy)/(w[1]**2)
                    W = np.where(r2 <= cutoff2, Cw * np.exp(-0.5 * r2), 0.0)
                    sl = np.ix_(ix, iy)

                    dv0 = vel[i, 0] - u[0][sl]
                    dv1 = vel[i, 1] - u[1][sl]
                    S[0][0][sl] += - mass[i] * dv0 * dv0 * W
                    S[0][1][sl] += - mass[i] * dv0 * dv1 * W
                    S[1][0][sl] += - mass[i] * dv1 * dv0 * W
                    S[1][1][sl] += - mass[i] * dv1 * dv1 * W
                else:
                    ix, iy, iz = idx_ranges
                    x = per_axis_centers[0]; y = per_axis_centers[1]; z = per_axis_centers[2]
                    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                    dx = X - pos[i, 0]; dy = Y - pos[i, 1]; dz = Z - pos[i, 2]
                    if periodic[0]: dx = _periodic_delta(dx, L[0])
                    if periodic[1]: dy = _periodic_delta(dy, L[1])
                    if periodic[2]: dz = _periodic_delta(dz, L[2])
                    r2 = (dx*dx)/(w[0]**2) + (dy*dy)/(w[1]**2) + (dz*dz)/(w[2]**2)
                    W = np.where(r2 <= cutoff2, Cw * np.exp(-0.5 * r2), 0.0)
                    sl = np.ix_(ix, iy, iz)

                    dv = [vel[i, a] - (u[a][sl]) for a in range(dim)]
                    for a in range(dim):
                        for b in range(dim):
                            S[a][b][sl] += - mass[i] * dv[a] * dv[b] * W

        # Contact: -½ Σ_ij f_ij ⊗ r_ij ∫_0^1 W(x - r_i - s r_ij) ds
        use_contacts = (contacts_i is not None) and (contacts_j is not None) and (contact_forces is not None)
        if use_contacts:
            contacts_i = np.asarray(contacts_i).reshape(-1)
            contacts_j = np.asarray(contacts_j).reshape(-1)
            contact_forces = np.asarray(contact_forces)
            if contact_forces.shape != (contacts_i.size, dim):
                raise ValueError("contact_forces must have shape (M, dim)")

            # --- remap global contact IDs to local indices; drop invalid pairs ---
            id2local = {int(pid): idx for idx, pid in enumerate(ids.tolist())}
            li = np.array([id2local.get(int(g), -1) for g in contacts_i], dtype=int)
            lj = np.array([id2local.get(int(g), -1) for g in contacts_j], dtype=int)
            valid = (li >= 0) & (li < N) & (lj >= 0) & (lj < N) & (li != lj)
            if not np.all(valid):
                li = li[valid]; lj = lj[valid]; contact_forces = contact_forces[valid]
            # (Optional) unique undirected pairs: keep i<j
            # keep = np.where(li < lj, True, False); li, lj, contact_forces = li[keep], lj[keep], contact_forces[keep]

            s_q, w_q = _leggauss_01(stress_quad)
            origin  = np.asarray(grid.origin, dtype=float)
            spacing = np.asarray(grid.spacing, dtype=float)
            shape   = np.asarray(grid.shape, dtype=int)

            for m in range(li.size):
                i = int(li[m]); j = int(lj[m])
                fij = contact_forces[m]              # force on i from j, shape (dim,)

                ri = pos[i].copy()
                rj = pos[j].copy()

                rij = rj - ri                        # branch vector (world)
                for a in range(dim):
                    if periodic[a]:
                        rij[a] = _periodic_delta(rij[a], L[a])

                # Bounding box around segment [ri, rj] expanded by cutoff_c * w
                seg_min = np.minimum(ri, ri + rij) - cutoff_c * w
                seg_max = np.maximum(ri, ri + rij) + cutoff_c * w

                # vectorized fractional index bounds (no int(np.floor(array)))
                lo_frac = (seg_min - origin) / spacing - 1.0
                hi_frac = (seg_max - origin) / spacing + 1.0
                lo_idx = np.maximum(np.floor(lo_frac).astype(int), 0)
                hi_idx = np.minimum(np.ceil (hi_frac).astype(int), shape - 1)

                idx_ranges = [np.arange(lo_idx[a], hi_idx[a] + 1, dtype=int) for a in range(dim)]
                per_axis_centers = [
                    origin[a] + (idx_ranges[a] + 0.5) * spacing[a] for a in range(dim)
                ]

                if dim == 2:
                    ix, iy = idx_ranges
                    x = per_axis_centers[0]; y = per_axis_centers[1]
                    X, Y = np.meshgrid(x, y, indexing='ij')
                    Wint = np.zeros((len(ix), len(iy)))
                    for q in range(len(s_q)):
                        rq = ri + s_q[q] * rij  # point along branch
                        dx = X - rq[0]; dy = Y - rq[1]
                        if periodic[0]: dx = _periodic_delta(dx, L[0])
                        if periodic[1]: dy = _periodic_delta(dy, L[1])
                        r2 = (dx*dx)/(w[0]**2) + (dy*dy)/(w[1]**2)
                        Wint += w_q[q] * np.where(r2 <= cutoff2, Cw * np.exp(-0.5 * r2), 0.0)
                    sl = np.ix_(ix, iy)
                    for a in range(dim):
                        for b in range(dim):
                            S[a][b][sl] += -0.5 * fij[a] * rij[b] * Wint
                else:
                    ix, iy, iz = idx_ranges
                    x = per_axis_centers[0]; y = per_axis_centers[1]; z = per_axis_centers[2]
                    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                    Wint = np.zeros((len(ix), len(iy), len(iz)))
                    for q in range(len(s_q)):
                        rq = ri + s_q[q] * rij
                        dx = X - rq[0]; dy = Y - rq[1]; dz = Z - rq[2]
                        if periodic[0]: dx = _periodic_delta(dx, L[0])
                        if periodic[1]: dy = _periodic_delta(dy, L[1])
                        if periodic[2]: dz = _periodic_delta(dz, L[2])
                        r2 = (dx*dx)/(w[0]**2) + (dy*dy)/(w[1]**2) + (dz*dz)/(w[2]**2)
                        Wint += w_q[q] * np.where(r2 <= cutoff2, Cw * np.exp(-0.5 * r2), 0.0)
                    sl = np.ix_(ix, iy, iz)
                    for a in range(dim):
                        for b in range(dim):
                            S[a][b][sl] += -0.5 * fij[a] * rij[b] * Wint

        stress_field = tuple(tuple(S[a][b] for b in range(dim)) for a in range(dim))

    # pack outputs
    scalars = {"occ": np.clip(occ, 0.0, 1.0)}
    if compute_scalars:
        if rho is not None: scalars["rho"] = rho
        if phi is not None: scalars["phi"] = phi

    vectors: Dict[str, Any] = {}
    if compute_vectors and pos_ref is not None:
        vectors["disp"] = disp
    if compute_vectors and vel is not None and compute_scalars:
        vectors["vel"] = vel_field

    tensors: Dict[str, Any] = {}
    if compute_stress and stress_field is not None:
        tensors["stress"] = stress_field

    return dict(
        scalars=scalars,
        vectors=vectors,
        tensors=tensors,
        params=dict(w_len=tuple(w.tolist()), cutoff_c=float(cutoff_c), periodic=tuple(periodic)),
    )
