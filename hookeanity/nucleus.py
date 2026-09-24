"""Exact body-frame OU increments with per-step co-rotating orientation updates."""

import json
from pathlib import Path
from typing import Any, Self, cast

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from . import h5
from .base import DIM, Real, VectorValues, instance
from .system import AndularGrid, Seed
from .system import System as SpatialSystem


def _transition_moments(
    drift: NDArray[np.floating],
    force: NDArray[np.floating],
    diffusion: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact transition matrix, additive mean, and covariance."""
    n = len(force)
    if dt == 0:
        return np.eye(n), np.zeros(n), np.zeros((n, n))

    norm = float(np.linalg.norm(drift, 1))
    if not np.isfinite(norm):
        raise ValueError("Nonfinite drift norm in exact OU transition")
    # Compute the logarithm before multiplying to avoid overflow in norm * dt.
    squarings = max(0, int(np.ceil(np.log2(norm) + np.log2(dt)))) if norm else 0
    h = np.ldexp(dt, -squarings)

    # At zero temperature only the affine deterministic exponential is needed.
    size = n + 1 if diffusion == 0 else 2 * n + 1
    block = np.zeros((size, size))
    block[:n, :n] = drift
    block[:n, -1] = force
    if diffusion != 0:
        block[:n, n : 2 * n] = diffusion * np.eye(n)
        block[n : 2 * n, n : 2 * n] = -drift.T
    exponential = expm(block * h)
    transition = exponential[:n, :n]
    mean = exponential[:n, -1]
    covariance = (
        np.zeros((n, n))
        if diffusion == 0
        else exponential[:n, n : 2 * n] @ transition.T
    )
    covariance = (covariance + covariance.T) / 2

    for _ in range(squarings):
        if diffusion != 0:
            covariance = covariance + transition @ covariance @ transition.T
            covariance = (covariance + covariance.T) / 2
        mean = mean + transition @ mean
        transition = transition @ transition

    if not all(np.isfinite(x).all() for x in (transition, mean, covariance)):
        raise ValueError("Nonfinite exact OU transition; check parameter magnitudes")
    return transition, mean, covariance


def _covariance_factor(covariance: np.ndarray) -> np.ndarray:
    """Factor a covariance, clipping only negative eigenvalues at roundoff scale."""
    symmetric = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = float(np.max(np.abs(eigenvalues)))
    tolerance = 100 * len(eigenvalues) * np.finfo(float).eps * scale
    if eigenvalues.min() < -tolerance:
        raise ValueError("Exact OU covariance is not positive semidefinite")
    return eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))


class System(SpatialSystem):
    """Elastic sphere with exact body-frame OU draws and finite rotation steps."""

    def __init__(
        self,
        resolution: tuple[int, int],
        lam: Real,
        mu: Real,
        dt: Real,
        f_ext: Real = 0.0,
        f_loc: tuple[Real, Real] = (0.0, 0.0),
        kbt: Real = 1.0,
        gamma: Real = 1.0,
        state: VectorValues | None = None,
        angular_grid: AndularGrid = "fibonacci",
        rng: np.random.Generator | Seed | None = None,
        verbose: bool = False,
        cache: str | Path | None = None,
        lmax: int | None = None,
        rotation: NDArray[np.floating] | None = None,
    ):
        """Initialize the spatial model and its exact transition.

        Args:
            resolution: Number of radial points and angular grid resolution.
            lam: First Lamé coefficient.
            mu: Shear modulus.
            dt: Nonnegative time interval between samples; zero is a no-op.
            f_ext: Constant external point-force magnitude.
            f_loc: Solid angle of the external point force.
            kbt: Nonnegative effective thermal energy.
            gamma: Positive dissipative coefficient.
            state: Initial point displacements, shape ``(npoints, 3)``.
            angular_grid: Fibonacci or Gauss–Legendre–Fourier angular grid.
            rng: NumPy generator or integer seed.
            verbose: Enable initialization logging.
            cache: Optional spatial operator cache.
            lmax: Spherical-harmonic expansion cutoff, or automatic selection.
            rotation: Body-to-lab orientation matrix; defaults to identity.

        Raises:
            ValueError: For invalid parameters, state, or bulk reduction.
            AssertionError: If the reduced traction has an unstable mode.
        """
        if angular_grid not in {"fibonacci", "gauss-legendre-fourier"}:
            raise ValueError(f"Unknown angular grid: {angular_grid!r}.")
        self._rotation = self._validated_rotation(rotation)

        self._resolution = (int(resolution[0]), int(resolution[1]))
        self._angular_grid = angular_grid
        self._lam = float(lam)
        self._mu = float(mu)
        self._dt = float(dt)
        self._ext = float(f_ext)
        self._loc = (float(f_loc[0]), float(f_loc[1]))
        self._kbt = float(kbt)
        self._gamma = float(gamma)
        self._lmax = None if lmax is None else int(lmax)
        parameters = [
            self._lam,
            self._mu,
            self._dt,
            self._ext,
            *self._loc,
            self._kbt,
            self._gamma,
        ]
        if not np.isfinite(parameters).all():
            raise ValueError("Physical parameters must be finite")
        if self._dt < 0 or self._gamma <= 0 or self._kbt < 0:
            raise ValueError("Require dt >= 0, gamma > 0, and kbt >= 0")
        self._rng = (
            rng
            if instance(rng, np.random.Generator)
            else np.random.default_rng(rng if instance(rng, Seed) else None)
        )
        self.verbose = verbose

        self.info("Initializing exact OU system")
        if cache is None:
            self._build_operators()
            self._build_rotation_projection()
            self._build_transition()
        else:
            self.load_cache(cache)
        self._state = (
            np.zeros((self.npoints, DIM))
            if state is None
            else np.array(state, dtype=float)
        )
        if self._state.shape != (self.npoints, DIM):
            raise ValueError(f"Expected state shape {(self.npoints, DIM)}")
        if not np.isfinite(self._state).all():
            raise ValueError("Initial state must be finite")
        self.info("System ready")

    def _build_transition(self):
        """Validate the reduced drift and refresh all time-dependent operators."""
        if not np.isfinite(self._trac_red).all() or not np.isfinite(self._force).all():
            raise ValueError("Surface traction and force must be finite")
        max_eigenvalue = np.real(np.linalg.eigvals(self._trac_red)).max()
        assert max_eigenvalue <= 1e-12, (
            f"Unstable mode detected ({max_eigenvalue:.3e}): increase lmax parameter"
        )
        self.info("  exact OU transition")
        self._transition, self._mean, self._covariance = _transition_moments(
            self._trac_red / self._gamma,
            self._force / self._gamma,
            2 * (self._kbt / self._gamma),
            self._dt,
        )
        self._noise_factor = (
            _covariance_factor(self._covariance)
            if self._kbt > 0 and self._dt > 0
            else np.zeros_like(self._covariance)
        )

        # The force is specified in lab coordinates. Integrate the response to
        # each lab-to-body matrix entry once, then combine at the step's start.
        force = self._force.reshape(-1, DIM)
        basis = np.zeros((self.nsurf, DIM * DIM))
        for lab_axis in range(DIM):
            for body_axis in range(DIM):
                basis[body_axis::DIM, DIM * lab_axis + body_axis] = (
                    force[:, lab_axis] / self._gamma
                )
        if self._dt == 0:
            self._force_response = np.zeros_like(basis)
        else:
            augmented = np.zeros((self.nsurf + DIM * DIM, self.nsurf + DIM * DIM))
            augmented[: self.nsurf, : self.nsurf] = self._trac_red / self._gamma
            augmented[: self.nsurf, self.nsurf :] = basis
            self._force_response = expm(augmented * self._dt)[
                : self.nsurf, self.nsurf :
            ]

    def __call__(self) -> Self:
        """Draw one frozen-frame OU step and compose its stochastic rotation."""
        if self._dt == 0:
            return self
        surf = (
            self._transition @ self._surface_state()
            + self._force_response @ self._rotation.ravel()
        )
        if self._kbt > 0:
            noise = self._noise_factor @ self._rng.normal(size=self.nsurf)
            internal_noise, omega = self._split_noise(noise)
            surf += internal_noise
            self._rotation = self._rotation @ Rotation.from_rotvec(omega).as_matrix()
        self._state = self._reconstruct_state(surf)
        return self

    def load_cache(self, filepath: str | Path) -> Self:
        """Load compatible spatial operators and rebuild the exact transition."""
        super().load_cache(filepath)
        self._build_rotation_projection()
        self._build_transition()
        return self

    def save(self, filepath: str | Path) -> Self:
        """Save full displacements, physical parameters, and RNG state to HDF5.

        The checkpoint contains the body-frame state and its lab orientation.
        """

        def json_default(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            raise TypeError(
                f"Object of type {type(obj).__name__} is not JSON serializable"
            )

        with h5py.File(filepath, "w") as f:
            f.create_dataset("points", data=self._grid, compression="lzf", shuffle=True)
            f.create_dataset("state", data=self._state, compression="lzf", shuffle=True)
            f.create_dataset("rotation", data=self._rotation)
            f.attrs.update(
                {
                    "resolution": self._resolution,
                    "gamma": self._gamma,
                    "lambda": self._lam,
                    "mu": self._mu,
                    "dt": self._dt,
                    "f_ext": self._ext,
                    "f_loc": self._loc,
                    "lmax": h5.optional_nnint(self._lmax),
                    "kT": self._kbt,
                    "angular_grid": self._angular_grid,
                    "rng_bit_generator": type(self._rng.bit_generator).__name__,
                    "rng_state_json": json.dumps(
                        self._rng.bit_generator.state, default=json_default
                    ),
                }
            )
        return self

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        verbose: bool = False,
        cache: str | Path | None = None,
    ) -> Self:
        """Load a checkpoint.

        Current ``system.System`` checkpoints are accepted; their method metadata
        is ignored. An optional spatial cache avoids rebuilding spatial operators.
        """

        def decode_text(value: Any) -> str:
            if isinstance(value, (bytes, bytearray, np.bytes_)):
                return bytes(value).decode("utf-8")
            if isinstance(value, str):
                return value
            raise ValueError(f"Expected text attribute, got {type(value).__name__}")

        def restore_rng(value: Any) -> Any:
            if isinstance(value, list):
                return np.asarray(value)
            if isinstance(value, dict):
                return {k: restore_rng(v) for k, v in value.items()}
            return value

        with h5py.File(filepath, "r") as f:
            name = decode_text(h5.attr_any(f, "rng_bit_generator"))
            bitgen_cls = getattr(np.random, name, None)
            if bitgen_cls is None:
                raise ValueError(f"Unknown NumPy bit generator: {name!r}")
            if not isinstance(bitgen_cls, type) or not issubclass(
                bitgen_cls, np.random.BitGenerator
            ):
                raise TypeError(f"Invalid NumPy bit generator: {name!r}")
            bitgen = bitgen_cls()
            bitgen.state = restore_rng(
                json.loads(decode_text(h5.attr_any(f, "rng_state_json")))
            )
            return cls(
                resolution=h5.attr_int_pair(f, "resolution"),
                lam=h5.attr_float(f, "lambda"),
                mu=h5.attr_float(f, "mu"),
                dt=h5.attr_float(f, "dt"),
                f_ext=h5.attr_float(f, "f_ext"),
                f_loc=h5.attr_float_pair(f, "f_loc"),
                kbt=h5.attr_float(f, "kT"),
                gamma=h5.attr_float(f, "gamma"),
                state=h5.dataset_ndfloat(f, "state"),
                angular_grid=cast(
                    AndularGrid, decode_text(h5.attr_any(f, "angular_grid"))
                ),
                rng=np.random.Generator(bitgen),
                verbose=verbose,
                cache=cache,
                lmax=h5.attr_optional_nnint(f, "lmax"),
                rotation=h5.dataset_ndfloat(f, "rotation"),
            )
