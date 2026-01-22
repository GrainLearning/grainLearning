import numpy as np

def check_mass_momentum_conservation(
    grid,
    *,
    pos, mass,
    vel=None,
    cg_out=None,
    atol_mass=1e-10,
    rtol_mass=1e-6,
    rtol_mom=1e-4,
):
    """
    Compare particle totals vs coarse-grained field integrals.

    - Mass:  sum(m_i)  ?=  ∑_cells rho * cell_vol
    - Momentum: sum(m_i v_i)  ?=  ∑_cells (rho * u) * cell_vol   (if vel available)

    Returns dict with absolute/relative errors and pass/fail flags.
    """
    mass = np.asarray(mass).reshape(-1)
    M_particles = mass.sum()
    if cg_out is None:
        raise ValueError("cg_out (result of coarse_grain) is required")

    scalars = cg_out.get("scalars", {})
    vectors = cg_out.get("vectors", {})
    rho = scalars.get("rho", None)
    vel_field = vectors.get("vel", None)  # tuple of arrays or None

    if rho is None:
        raise ValueError("cg_out['scalars']['rho'] missing")

    # Integrate mass on grid
    M_grid = float(rho.sum() * grid.cell_vol)

    mass_abs_err = abs(M_grid - M_particles)
    mass_rel_err = mass_abs_err / max(M_particles, 1e-16)

    out = {
        "mass_particles": M_particles,
        "mass_grid": M_grid,
        "mass_abs_err": mass_abs_err,
        "mass_rel_err": mass_rel_err,
        "mass_ok": (mass_abs_err <= atol_mass) or (mass_rel_err <= rtol_mass),
    }

    # Momentum (only if vel provided and coarse-grained velocity exists)
    if vel is not None and vel_field is not None:
        vel = np.asarray(vel)
        P_particles = vel.T @ mass  # shape (dim,)

        # ∑ rho * u * dV
        dim = vel.shape[1]
        P_grid = np.zeros(dim, dtype=float)
        for a in range(dim):
            P_grid[a] = float((rho * vel_field[a]).sum() * grid.cell_vol)

        mom_abs_err = np.linalg.norm(P_grid - P_particles)
        mom_rel_err = mom_abs_err / max(np.linalg.norm(P_particles), 1e-16)

        out.update({
            "momentum_particles": P_particles,
            "momentum_grid": P_grid,
            "momentum_abs_err": mom_abs_err,
            "momentum_rel_err": mom_rel_err,
            "momentum_ok": (mom_rel_err <= rtol_mom),
        })
    else:
        out.update({
            "momentum_particles": None,
            "momentum_grid": None,
            "momentum_abs_err": None,
            "momentum_rel_err": None,
            "momentum_ok": None,
        })

    return out
