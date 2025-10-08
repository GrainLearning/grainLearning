"""
High-level ROM pipeline classes that abstract away normalization and internal details.

Users can simply instantiate a ROM class, fit it on data, and predict - all normalization
is handled transparently under the hood.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple, Dict, Any

from rom_pod_ae import (
    build_snapshots_from_list, center_snapshots, pod, train_autoencoder,
    build_master_pod, build_master_snapshots, transform, inverse_transform
)

from rom_sindy_gp import (
    fit_sindy_continuous, simulate_and_reconstruct, simulate_and_reconstruct_gp,
    simulate_and_reconstruct_autoencoder, fit_sindycp_continuous, simulate_and_reconstruct_cp,
    multivariate_GP
)
from rom_io import load_2d_trajectory_from_file, print_error_metrics, plot_ae_history


class BaseROM(ABC):
    """Abstract base class for all ROM pipelines."""
    
    def __init__(self, normalization: bool = True, energy: float = 0.99, tmax = None):
        """
        Parameters:
        - normalization: Whether to apply channel-wise min-max normalization [0,1]
        - energy: POD energy retention threshold
        - tmax: Maximum time index for loading data, Default: None (load all)
        """
        self.normalization = normalization
        self.energy = energy
        self.tmax = tmax
        self.channels = None  # set after fitting on training data
        self.channel_bounds = None  # set after fitting on training data
        self.is_fitted = False
        self.tag = "ROM"
        
        # Internal state (set during fit)
        self.A_train = None
        self.U_r_train = None
        self.xbar_train = None
        self.shape = None
        self.num_modes = None

        # Error metrics
        self.global_error = None
        self.errors = None
        
    def initialize_from_file(self, file: str, channels: List[str], vec_field_ids: Optional[List[int]] = None, t_max: Optional[int] = None):
        """Load channel list from a .npy file."""
        self.tmax = t_max
        self.channels = channels
        self.channel_list = load_2d_trajectory_from_file(file, channels=channels, t_max=t_max)
        self.vec_field_ids = vec_field_ids

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the ROM on training data."""
        pass
        
    @abstractmethod
    def predict(self, *args, **kwargs):
        """Generate predictions from the fitted ROM."""
        pass
        
    def _prepare_training_snapshots(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Build snapshots for training: computes and stores channel_bounds when normalize=True."""
        X, shape, bounds = build_snapshots_from_list(self.channel_list, normalization=self.normalization)
        self.channel_bounds = bounds
        return X, shape
        
    def _prepare_testing_snapshots(self, channel_list: List[np.ndarray]):
        X_test_orig, _, _ = build_snapshots_from_list(channel_list, normalization=False)
        if self.normalization and self.channel_bounds is None:
            raise RuntimeError("Channel bounds are not set although normalization is enabled.")
        elif not self.normalization and self.channel_bounds is None:
            return X_test_orig
        else:
            return transform(X_test_orig, self.channel_bounds)

    def _unnormalize_predictions(self, X_pred: np.ndarray) -> np.ndarray:
        """Internal helper to unnormalize predictions if needed."""
        if self.normalization and self.channel_bounds is not None:
            return inverse_transform(X_pred, self.channel_bounds)
        return X_pred

    def _unpack_snapshots(self, X: np.ndarray) -> np.ndarray:
        """Unpack a concatenated snapshot into an array of size (C, nx, ny, T)."""
        from rom_io import unpack_2d_field
        num_channels = len(self.channel_bounds) if self.channel_bounds is not None else 1
        return unpack_2d_field(X, self.shape, list(range(num_channels)))
    
    def _build_pod(self, X):
        # Center and POD
        Xc, self.xbar_train = center_snapshots(X)
        U_r, A, _ = pod(Xc, energy=self.energy)
        
        # Truncate modes
        r_full = U_r.shape[1]
        self.num_modes = min(self.num_modes, r_full)
        self.U_r_train = U_r[:, :self.num_modes]
        self.A_train = A[:, :self.num_modes]        
        print(f"[POD] kept r = {r_full} modes (energy {self.energy*100:.0f}%)")

    def evaluate(self, data_or_file: Union[str, List[np.ndarray]], t: np.ndarray, create_visual: bool = False, every: int = 10):
        """
        Evaluate ROM accuracy on test data.
        
        Parameters:
        - data_or_file: Either a .npy file path (str) or a preloaded list of channel arrays
        - t: Time vector matching the number of snapshots
        - tag: Tag for error printing
        """
        if not self.is_fitted:
            raise RuntimeError("ROM must be fitted before evaluation.")
        # Accept either a file path or a preloaded list
        if isinstance(data_or_file, str):
            t_max = t.shape[0]
            channel_list = load_2d_trajectory_from_file(data_or_file, channels=self.channels, t_max=t_max)
        else:
            channel_list = data_or_file
        self.X_test = self._prepare_testing_snapshots(channel_list)

        T = self.X_test.shape[1]
        if len(t) != T:
            raise ValueError("Length of provided t does not match test snapshots T.")
        t_query = np.asarray(t)
        self.X_pred = self.predict(t_query)
        
        self.global_error, self.errors = print_error_metrics(self.X_test, self.X_pred, tag=self.tag)
        if create_visual:
            self.visualize(t, tag=self.tag, every=every)

    def visualize(self, t: np.ndarray, tag: str = "ROM", every: int = 10):
        """
        Create visualizations and GIFs for ROM predictions.
        
        Parameters:
        - t: Time vector matching the number of snapshots
        - tag: Tag for file naming
        - every: Save every nth snapshot for the visualization
        """
        if not self.is_fitted:
            raise RuntimeError("ROM must be fitted before visualization.")
            
        from rom_io import visualize_2d_field, visualize_2d_field_magnitude, create_gif_from_pngs
               
        # Visualize the 2D field over time
        for i in range(0, len(t), every):
            for j, c in enumerate(self.channels):
                name = f'{c}_field'
                visualize_2d_field(
                    inverse_transform(self.X_test, self.channel_bounds),
                    inverse_transform(self.X_pred, self.channel_bounds),
                    self.shape, time_index=i, channel=j,
                    name=name, tag=tag)
                create_gif_from_pngs(name=f'{tag}_{name}')
            if self.vec_field_ids is not None and len(self.vec_field_ids) == 2:
                # extract vector field name
                vec_field_name = self.channels[self.vec_field_ids[0]]
                vec_field_name = vec_field_name.split('_')[0]  # e.g., 'vel' from 'vel_x'
                name = f'{vec_field_name}_field_magnitude'
                visualize_2d_field_magnitude(
                    inverse_transform(self.X_test, self.channel_bounds),
                    inverse_transform(self.X_pred, self.channel_bounds),
                    self.shape, time_index=i, channel=self.vec_field_ids,
                    name=name, tag=tag)

    @abstractmethod
    def reconstruct_training(self) -> np.ndarray:
        """Reconstruct the training dataset using the fitted model components.

        Returns original-scale snapshot matrix (D, T_train).
        """
        pass


class PodGpROM(BaseROM):
    """POD + Gaussian Process ROM pipeline."""
    
    def __init__(self, normalization: bool = True, energy: float = 0.99, num_modes: int = 10,
                 tag: str = "POD-GP"):
        super().__init__(normalization, energy)
        self.num_modes = num_modes
        self.t_train = None
        self.tag = tag
        
    def fit(self, file: str, channels: List[str], dt: float, t_max: Optional[int] = None,
        vec_field_ids: Optional[List[int]] = None) -> 'PodGpROM':
        """
        Fit a POD-GP ROM
        
        Parameters:
        - file: .npy file path
        - channels: List of channel names to load
        - dt: Time step size
        - t_max: Maximum time index for loading data, Default: None (load all)
        - vec_field_ids: Indices of vector field channels, e.g., [1,2] for ['vel_x', 'vel_y']
        
        Returns:
        - self (for method chaining)
        """
        # Load data
        self.initialize_from_file(file, channels, vec_field_ids, t_max)
        X, self.shape = self._prepare_training_snapshots()
        
        # Build POD
        self._build_pod(X)

        self.is_fitted = True

        # Store training time
        self.t_train = np.arange(0, X.shape[1] * dt, dt)
        return self
        
    def predict(self, t_query: np.ndarray) -> np.ndarray:
        """
        Predict field evolution at query times using GP.
        
        Parameters:
        - t_query: Query time points
        
        Returns:
        - X_pred: Predicted fields (D, T_query) in original physical units
        """
        if not self.is_fitted:
            raise RuntimeError("ROM must be fitted before prediction.")
            
        X_pred = simulate_and_reconstruct_gp(
            self.U_r_train, self.A_train, self.t_train, t_query,
            xbar=self.xbar_train
        )
        return X_pred

    def reconstruct_training(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ROM must be fitted before reconstruction.")
        if self.A_train is None:
            raise RuntimeError("Training coefficients not stored.")
        Xn = (self.U_r_train @ self.A_train.T)
        if self.xbar_train is not None:
            Xn = Xn + self.xbar_train[:, None]
        return self._unnormalize_predictions(Xn)


class PodSindyROM(BaseROM):
    """POD + SINDy ROM pipeline."""
    
    def __init__(self, normalization: bool = True, energy: float = 0.99, num_modes: int = 3, 
                 poly_degree: int = 2, thresh: float = 0.1, diff: str = "smoothed",
                 tag: str = "POD-SINDY"):
        super().__init__(normalization, energy)
        self.num_modes = num_modes
        self.poly_degree = poly_degree
        self.thresh = thresh
        self.diff = diff
        self.model = None
        self.tag = tag
        
    def fit(self, file: str, channels: List[str], dt: float, t_max: Optional[int] = None,
             vec_field_ids: Optional[List[int]] = None) -> 'PodSindyROM':
        """Fit a POD-SINDy ROM

        Parameters:
        - file: .npy file path
        - channels: List of channel names to load
        - dt: Time step size
        - t_max: Maximum time index for loading data, Default: None (load all)
        - vec_field_ids: Indices of vector field channels, e.g., [1,2] for ['vel_x', 'vel_y']

        Returns:
        - self (for method chaining)
        """
        # Load data
        self.initialize_from_file(file, channels, vec_field_ids, t_max)
        X, self.shape = self._prepare_training_snapshots()
        
        # Build POD
        self._build_pod(X)

        # Fit SINDy with user-provided time sequence
        self.t_train = np.arange(0, X.shape[1] * dt, dt)
        self.model = fit_sindy_continuous(
            self.A_train, self.t_train, 
            poly_degree=self.poly_degree, thresh=self.thresh, diff=self.diff)
        self.model.print()

        # Store initial condition
        self.A0 = self.A_train[0, :]
        self.is_fitted = True
        return self

    def predict(self, t_eval: np.ndarray, A0: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict field evolution using SINDy rollout.

        Returns normalized predictions to match BaseROM.evaluate behavior.
        """
        if not self.is_fitted:
            raise RuntimeError("ROM must be fitted before prediction.")

        if A0 is None:
            A0 = self.A0
        X_pred = simulate_and_reconstruct(self.model, self.U_r_train, A0, t_eval, xbar=self.xbar_train)
        return X_pred

    def reconstruct_training(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ROM must be fitted before reconstruction.")
        if self.A_train is None:
            raise RuntimeError("Training coefficients not stored.")
        Xn = (self.U_r_train @ self.A_train.T)
        if self.xbar_train is not None:
            Xn = Xn + self.xbar_train[:, None]
        return self._unnormalize_predictions(Xn)


class AutoencoderSindyROM(BaseROM):
    """Autoencoder + SINDy ROM pipeline."""
    
    def __init__(self, normalization: bool = True, latent_dim: int = 3, 
                 poly_degree: int = 2, thresh: float = 1.0, diff: str = "smoothed",
                 epochs: int = 1000, batch_size: int = 64, val_split: float = 0.2,
                 patience: int = 200, print_every: int = 50, device: str = "cuda",
                 tag: str = "AE-SINDY"):
        super().__init__(normalization, energy=1.0)  # AE doesn't use POD energy
        self.latent_dim = latent_dim
        self.poly_degree = poly_degree
        self.thresh = thresh
        self.diff = diff
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.patience = patience
        self.print_every = print_every
        self.device = device
        # Model components
        self.encoder = None
        self.decoder = None
        self.model = None
        self.A0 = None
        self.tag = tag
        
    def fit(self, file: str, channels: List[str], dt: float, t_max: Optional[int] = None,
             vec_field_ids: Optional[List[int]] = None) -> 'AutoencoderSindyROM':
        """Fit a Autoencoder-SINDy ROM

        Parameters:
        - file: .npy file path
        - channels: List of channel names to load
        - dt: Time step size
        - t_max: Maximum time index for loading data, Default: None (load all)
        - vec_field_ids: Indices of vector field channels, e.g., [1,2] for ['vel_x', 'vel_y']

        Returns:
        - self (for method chaining)
        """
        # Load data
        self.initialize_from_file(file, channels, vec_field_ids, t_max)
        X, self.shape = self._prepare_training_snapshots()

        # Train autoencoder
        self.encoder, self.decoder, self.A_train, history = train_autoencoder(
            X, latent_dim=self.latent_dim, epochs=self.epochs, batch_size=64, lr=1e-4,
            device=self.device, val_split=0.2, patience=200, print_every=50
        )

        # Save training history plot
        plot_ae_history(history, savepath=f"{self.tag}_train_val_loss.png")

        # Fit SINDy on latent coefficients with provided time vector
        self.t_train = np.arange(0, X.shape[1] * dt, dt)
        self.model = fit_sindy_continuous(
            self.A_train, self.t_train, 
            poly_degree=self.poly_degree, thresh=self.thresh, diff=self.diff)
        self.model.print()

        # Store initial condition
        self.A0 = self.A_train[0, :]
        self.is_fitted = True
        return self

    def predict(self, t_eval: np.ndarray, A0: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict field evolution using AE + SINDy rollout.

        Returns normalized predictions to match BaseROM.evaluate behavior.
        """
        if not self.is_fitted:
            raise RuntimeError("ROM must be fitted before prediction.")

        if A0 is None:
            A0 = self.A0

        X_pred = simulate_and_reconstruct_autoencoder(
            self.model, self.decoder, A0, t_eval,
            device=self.device
        )
        return X_pred

    def reconstruct_training(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ROM must be fitted before reconstruction.")
        if self.decoder is None or self.A_train is None:
            raise RuntimeError("Autoencoder not trained or latent codes missing.")
        # Decode the latent trajectory back to snapshot space
        import importlib
        torch = importlib.import_module('torch')
        with torch.no_grad():
            A_tensor = torch.tensor(self.A_train, dtype=torch.float32).to(self.device)
            Xn_T = self.decoder(A_tensor).cpu().numpy()  # shape (T, D)
        Xn = Xn_T.T  # (D, T)
        return self._unnormalize_predictions(Xn)


class ParametricPodSindyROM(PodSindyROM):
    """Parametric POD + SINDy-CP ROM for parameteric systems."""

    def __init__(self, normalization: bool = True, energy: float = 0.99, num_modes: int = 3,
                 poly_deg_state: int = 1, poly_deg_param: int = 1, thresh: float = 0.1,
                 diff: str = "smoothed", tag: str = "POD-SINDy_Param"):
        # Initialize like a POD-SINDY model; poly_degree here mirrors state degree
        super().__init__(normalization=normalization, energy=energy, num_modes=num_modes,
                         poly_degree=poly_deg_state, thresh=thresh, diff=diff, tag=tag)
        # Parametric-specific settings
        self.poly_deg_state = poly_deg_state
        self.poly_deg_param = poly_deg_param
        self.u_list_train = None  # List of parameter vectors per training run
        self.A_list_train = None  # List of POD coeffs per training run
        self.gp_initial = None  # GP for predicting initial conditions

    def fit(self, file_list: List[str], u_list: List[np.ndarray], channels: List[str],
            dt: float, t_max: Optional[int] = None,
            vec_field_ids: Optional[List[int]] = None) -> 'ParametricPodSindyROM':
        """
        Fit a parametric POD-SINDY ROM
        
        Parameters:
        - file_list: List of .npy file paths
        - u_list: List of parameter vectors per run
        - channels: List of channel names to load
        - dt: Time step size
        - t_max: Optional cap on time steps per run (None means use all)
        - vec_field_ids: Indices of vector field channels, e.g., [1,2] for ['vel_x', 'vel_y']
        
        Returns:
        - self (for method chaining)
        """
        self.u_list_train = u_list
        self.vec_field_ids = vec_field_ids
        # Build master snapshots with simple normalization and store bounds
        self.channels = channels
        X_concat, shapes, X_list, channel_bounds = build_master_snapshots(
            file_list, channels=channels, t_max=t_max, normalization=self.normalization
        )
        self.channel_bounds = channel_bounds
        self.U_r_train, A_list_full = build_master_pod(X_concat, X_list, energy=self.energy)
        r_full = self.U_r_train.shape[1]
        print(f"[POD] train: kept r = {r_full} modes (energy {self.energy*100:.1f}%)")
        self.shape = shapes[0]  # Assume all same shape
        
        # Truncate modes
        r_full = self.U_r_train.shape[1]
        self.num_modes = min(self.num_modes, r_full)
        self.A_list_train = [A[:, :self.num_modes] for A in A_list_full]
        self.U_r_train = self.U_r_train[:, :self.num_modes]

        # Fit SINDy-CP exactly like pod_sindy_param_driver
        self.t_train = np.arange(0, X_list[0].shape[1] * dt, dt)  # assume uniform dt and length
        self.model = fit_sindycp_continuous(
            A_list=self.A_list_train,
            t_list=self.t_train,
            u_list=self.u_list_train,
            poly_deg_state=self.poly_deg_state,
            poly_deg_param=self.poly_deg_param,
            thresh=self.thresh,
            diff=self.diff
        )
        self.model.print()
        
        # Fit GP for initial condition prediction
        A0_list_train = [A[0] for A in self.A_list_train]
        self.gp_initial = multivariate_GP()
        self.gp_initial.fit(self.u_list_train, A0_list_train)

        self.is_fitted = True
        return self
        
    def predict(self, u_list_new: List[np.ndarray], t_eval: np.ndarray) -> List[np.ndarray]:
        """
        Predict field evolution for new parameter configuration.
        
        Parameters:
        - u_list_new: List of new parameter vectors (one per run)
        - t_eval: Time points for evaluation
        
        Returns:
        - X_list_pred: List of predicted fields (D, T_eval)
        """
        if not self.is_fitted:
            raise RuntimeError("ROM must be fitted before prediction.")

        # Predict initial condition
        A0_pred = self.gp_initial.predict(u_list_new)

        # Rollout with SINDy-CP
        X_list_pred = simulate_and_reconstruct_cp(
            self.model, self.U_r_train, A0_pred, t_eval=t_eval, u=u_list_new
        )
        return X_list_pred
        
    def evaluate_parametric(self, file_list: List[str], u_list_new: List[np.ndarray], t: np.ndarray,
                            create_visual: bool = False, every: int = 10):
        """
        Evaluate the trained parametric ROM on new runs

        Parameters:
        - file_list: List of .npy file paths
        - u_list_new: List of parameter vectors per run
        - t: Time vector matching the number of snapshots
        - create_visual: Whether to create visualizations/GIFs
        - every: Save every nth snapshot for the visualization
        """
        from rom_io import print_global_error
        
        # Throw an error if model is not fitted
        if not self.is_fitted:
            raise RuntimeError("ROM must be fitted before evaluation.")

        # Get training data snapshots (original scale)
        t_max = t.shape[0]
        _, _, X_list_new, _ = build_master_snapshots(
            file_list, channels=self.channels, t_max=t_max, normalization=False
        )
        # Use training bounds to normalize new data if needed
        X_list_new = [transform(X, self.channel_bounds) for X in X_list_new]

        # Evaluate each run
        X_list_pred = self.predict(u_list_new, t)
        
        # Evaluate error (and create visualization) for each run
        avg_error = 0.0
        errors = []
        for i, (X_new, X_pred) in enumerate(zip(X_list_new, X_list_pred, u_list_new)):
            error = print_global_error(X_new, X_pred, tag=f'{self.tag} [Sample_{i:<02d}]')
            errors.append(error)
            avg_error += error
            # Save evolution of the 2D field into a GIF
            if create_visual:
                self.visualize(t, tag=self.tag, every=every)
        avg_error = sum(errors) / len(errors)
        print(f"{self.tag} Average training error: {avg_error:.4f}")
        return errors


# Convenience factory functions
def create_pod_gp_rom(**kwargs) -> PodGpROM:
    """Create a POD+GP ROM with sensible defaults."""
    return PodGpROM(**kwargs)

def create_pod_sindy_rom(**kwargs) -> PodSindyROM:
    """Create a POD+SINDy ROM with sensible defaults."""
    return PodSindyROM(**kwargs)

def create_autoencoder_sindy_rom(**kwargs) -> AutoencoderSindyROM:
    """Create an Autoencoder+SINDy ROM with sensible defaults."""
    return AutoencoderSindyROM(**kwargs)

def create_parametric_sindy_rom(**kwargs) -> ParametricPodSindyROM:
    """Create a Parametric POD+SINDy ROM with sensible defaults."""
    return ParametricPodSindyROM(**kwargs)


# ------------------------------
# Unified, task-oriented API
# ------------------------------
def run_rom_pipeline(
    tag: str,
    task: str,
    # Single-run data
    file: Optional[str] = None,
    channels: Optional[List[str]] = None,
    t: Optional[np.ndarray] = None,
    # Parametric data (multi-run)
    file_list: Optional[List[str]] = None,
    u_list: Optional[List[np.ndarray]] = None,
    # Optional parametric test set
    file_list_test: Optional[List[str]] = None,
    u_list_test: Optional[List[np.ndarray]] = None,
    # Common options
    create_visual: bool = False,
    normalization: bool = True,
    # POD/GP/SINDy params
    energy: float = 0.99,
    num_modes: int = 10,
    poly_degree: int = 1,
    thresh: float = 0.1,
    diff: str = "smoothed",
    # AE params
    latent_dim: int = 2,
    ae_epochs: int = 5000,
    device: str = "cuda",
    # data params
    t_max: Optional[int] = None,
    train_ratio: float = 0.8,
) -> Dict[str, Any]:
    """
    Unified driver that selects a ROM by tag and performs the requested task.

    tag: 'POD-GP' | 'POD-SINDY' | 'AE-SINDY' | 'POD-SINDY-Param'
    task: 'reconstruction' | 'training' | 'testing'
    """
    tag_upper = tag.upper()

    # Instantiate the selected model
    if tag_upper == "POD-GP":
        rom = PodGpROM(normalization=normalization, energy=energy, num_modes=num_modes, tag=tag_upper)
    elif tag_upper == "POD-SINDY":
        rom = PodSindyROM(normalization=normalization, energy=energy, num_modes=num_modes,
                          poly_degree=poly_degree, thresh=thresh, diff=diff, tag=tag_upper)
    elif tag_upper == "AE-SINDY":
        rom = AutoencoderSindyROM(normalization=normalization, latent_dim=latent_dim,
                                  poly_degree=poly_degree, thresh=thresh, diff=diff,
                                  epochs=ae_epochs, device=device, tag=tag_upper)
    elif tag_upper == "POD-SINDY-PARAM":
        rom = ParametricPodSindyROM(normalization=normalization, energy=energy, num_modes=num_modes,
                                    poly_deg_state=poly_degree, poly_deg_param=poly_degree,
                                    thresh=thresh, diff=diff, tag=tag_upper)
    else:
        raise ValueError("tag must be one of: 'POD-GP', 'POD-SINDY', 'AE-SINDY', 'POD-SINDY-Param'")

    results: Dict[str, Any] = {"rom": rom}

    # Non-parametric branches: validate single-run inputs
    if tag_upper in {"POD-GP", "POD-SINDY", "AE-SINDY"}:
        if file is None or channels is None or t is None:
            raise ValueError("For non-parametric models, 'file', 'channels', and 't' must be provided.")
        # Load for sizing/splitting only
        U_list = load_2d_trajectory_from_file(file, channels=channels, t_max=t_max)
        T_total = U_list[0].shape[0]
        if len(t) != T_total:
            raise ValueError("Length of provided t does not match loaded data T.")
        t_full = np.asarray(t)
        # Compute dt (require uniform spacing)
        if len(t_full) < 2:
            raise ValueError("Time array 't' must have at least 2 points to compute dt.")
        diffs = np.diff(t_full)
        if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-9):
            raise ValueError(f"{tag_upper} requires uniform time spacing to compute dt from t.")
        dt = float(diffs[0])

        # Tasks
        if task == "reconstruction":
            rom.fit(file=file, channels=channels, dt=dt, t_max=t_max)
            X_rec = rom.reconstruct_training()
            results["reconstruction"] = X_rec
            return results

        elif task == "training":
            rom.fit(file=file, channels=channels, dt=dt, t_max=t_max)
            results["status"] = "trained"
            return results

        elif task == "testing":
            split_idx = int(T_total * train_ratio)
            # Test split
            U_test = [u[split_idx:] for u in U_list]
            t_train = t_full[:split_idx]
            t_test = t_full[split_idx:]
            # Fit on train portion by limiting t_max
            rom.fit(file=file, channels=channels, dt=dt, t_max=split_idx)
            # Evaluate on test portion (normalized metrics, original-scale visuals if enabled)
            rom.tag = f"[test] {tag_upper}"
            rom.evaluate(U_test, t_test, create_visual=create_visual)
            results["metrics"] = {
                "global_error": rom.global_error,
                "per_channel_errors": rom.errors,
            }
            return results
        else:
            raise ValueError("task must be one of: 'reconstruction', 'training', 'testing'")

    # Parametric branch
    else:
        if file_list is None or u_list is None or channels is None or t is None:
            raise ValueError("For 'POD-SINDY-Param', provide 'file_list', 'u_list', 'channels', and 't'.")
        t_full = np.asarray(t)
        if len(t_full) < 2:
            raise ValueError("Time array 't' must have at least 2 points to compute dt.")
        diffs = np.diff(t_full)
        if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-9):
            raise ValueError("POD-SINDY-Param requires uniform time spacing to compute dt from t.")
        dt = float(diffs[0])

        if task == "reconstruction":
            # Fit and return reconstructions of training runs
            rom.fit(file_list=file_list, u_list=u_list, channels=channels, dt=dt, t_max=t_max)
            # Reconstruct each training run from stored A_list_train
            X_rec_list = []
            for A in rom.A_list_train:
                Xn = rom.U_r_train @ A.T
                if rom.xbar_train is not None:
                    Xn = Xn + rom.xbar_train[:, None]
                X_rec_list.append(rom._unnormalize_predictions(Xn))
            results["reconstruction"] = X_rec_list
            return results

        elif task == "training":
            rom.fit(file_list=file_list, u_list=u_list, channels=channels, dt=dt, t_max=t_max)
            results["status"] = "trained"
            return results

        elif task == "testing":
            if file_list_test is None or u_list_test is None:
                raise ValueError("Provide 'file_list_test' and 'u_list_test' for parametric testing.")
            # Fit on training runs
            rom.fit(file_list=file_list, u_list=u_list, channels=channels, dt=dt, t_max=t_max)
            # Evaluate on separate test runs
            per_run_errors = rom.evaluate_parametric(file_list=file_list_test, u_list_new=u_list_test, t=t_full,
                                                    create_visual=create_visual)
            # Store for convenience
            rom.errors = per_run_errors
            results["metrics"] = {
                "per_run_errors": per_run_errors
            }
            return results
        else:
            raise ValueError("task must be one of: 'reconstruction', 'training', 'testing'")