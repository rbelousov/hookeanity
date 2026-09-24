"""Spherical body subject to fluctuating forces and point-source surface forces.

Attributes:
    Seed (typing.TypeAlias): Random-number generator seed.
    Init (typing.TypeAlias): Option for traction operator correction ("no_rigid_motion", "no_translations", "no_rotations").
    IntegrationMethod (typing.TypeAlias): Method of time integration ("euler", "trotter", "strang").
    AndularGrid (typing.TypeAlias): Type of the solid-angle grid ("fibonacci", "gauss-legendre-fourier").
"""

from pathlib import Path
from typing import cast, Any, Literal, Self, TypeAlias
import h5py
import json
import numpy as np
from scipy.sparse.linalg import expm_multiply, LinearOperator
from scipy.spatial import ConvexHull
from numpy.typing import NDArray
from .interpolation import Interpolation
from .base import DIM, Real, Grid, VectorValues, instance
from .logging import logger
from . import h5

Seed: TypeAlias = int | np.integer
IntegrationMethod: TypeAlias = Literal["euler", "trotter", "strang"]
AndularGrid: TypeAlias = Literal["fibonacci", "gauss-legendre-fourier"]
Init: TypeAlias = Literal["no_rigid_motion", "no_translations", "no_rotations"] | None


def cast_init(x: Any) -> Init:
    """Convert object to `Init`."""
    return cast(Init, None if x == "none" else str(x))


class System:
    """Simulated system.

    Attributes:
        _resolution (tuple[int, int]): Number of radial points and angular resolution..
        _angular_grid (hookeanity.system.AndularGrid): Type of grid in the angular component ("fibonacci, gauss-legendre-fourier").
        _gamma (float): Dissipative coefficient.
        _lam (float): Lamé lambda coefficient.
        _mu (float): Lamé mu coefficient.
        _dt (float): Integration time step
        _ext (float): Magnitude of the external point force.
        _loc (tuple[float, float]): Solid angle specifying the location of the external point force.
        _lmax (int | None): Order of the spherical-harmonic expansion of the Sobolev-Matérn kernels.
        _kbt (float): Temperature.
        _rng (numpy.random.Generator): Numpy random-number generator.
        verbose (bool): Trigger of logging output.
        _state (hookeanity.VectorValues): Displacements of the reference points.
        _tf (float): Time step normalized by the dissipative coefficient.
        _method (hookeanity.system.IntegrationMethod): Time-integration method.
        _trac_oper (scipy.sparse.linalg.LinearOperator): Matrix traction operator for Trotter or Strang methods of time integration.
        _trace (hookeanity.Real): Trace of the matrix traction operator for Trotter or Strang methods of integration.
        _std (numpy.ndarray): Intensity of the thermal fluctuations.
        _mesh (tuple[int, int]): Number of points in the radial and solid-angle components.
        _grid (hookeanity.Grid): 3D mesh grid.
        _trac (numpy.ndarray): Matrix operator of the elastic traction force.
        _ela (numpy.ndarray): Matrix operator of the divergence of the elastic stress.
        _force (numpy.ndarray): Components of external point force over surface degrees of freedom.
        _trac_red (numpy.ndarray): Matrix traction operator reduced to the surface degrees of freedom.
    """

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
        method: IntegrationMethod = "euler",
        angular_grid: AndularGrid = "fibonacci",
        rng: np.random.Generator | Seed | None = None,
        verbose: bool = False,
        cache: str | Path | None = None,
        lmax: int | None = None,
        init: Init = "no_rigid_motion",
    ):
        """Initialize the simulation system.

        Args:
            resolution: Number of radial points and angular grid resolution.
            lam: Lamé's first parameter.
            mu: Lamé's second parameter (shear modulus).
            dt: Integration time step.
            f_ext: Magnitude of the external force defaults to zero.
            f_loc: Solid angle specifying the location of the external point force.
            kbt: Effective temperature in units of energy defaults to one.
            gamma: Effective dissipative coefficient.
            state: Current state of the system defaults to zeros.
            method: Integration method ("euler", "trotter", "strang").
            angular_grid: Type of grid in the angular component ("fibonacci, gauss-legendre-fourier").
            rng: Optional state of the random number generator or seed.
            verbose: Log the initialization.
            cache: Path to the cached numerical data.
            lmax: Order of the spherical-harmonic expansion of the Sobolev-Matérn kernels.
            init: Surface projection mode for reduced traction.

        Raises:
            ValueError: On unknown integration method.
            AssertionError: When traction operator has a positive eigenvalue.
        """
        if method not in {"euler", "strang", "trotter"}:
            raise ValueError(f"Unknown integration method: {method!r}.")
        if angular_grid not in {"fibonacci", "gauss-legendre-fourier"}:
            raise ValueError(f"Unknown angular grid: {angular_grid!r}.")
        if init not in {"no_rigid_motion", "no_translations", "no_rotations", None}:
            raise ValueError(f"Unknown init option: {init!r}.")

        self._resolution = (int(resolution[0]), int(resolution[1]))
        self._angular_grid = angular_grid
        self._gamma = float(gamma)
        self._lam = float(lam)
        self._mu = float(mu)
        self._dt = float(dt)
        self._ext = float(f_ext)
        self._loc = (float(f_loc[0]), float(f_loc[1]))
        self._lmax = None if lmax is None else int(lmax)
        self._init: Init = init
        self._kbt = float(kbt)
        self._method = method
        self._rng = (
            rng
            if instance(rng, np.random.Generator)
            else np.random.default_rng(rng if instance(rng, Seed) else None)
        )
        self.verbose = verbose

        # Initialization starts
        self.info("Initializing system")
        if cache is None:
            self._build_operators()
        else:
            self.info(f"  loading cached data from {cache}")
            self.load_cache(cache)
        self._state = (
            np.zeros((np.prod(self._mesh), DIM)) if state is None else np.array(state)
        )

        self._tf = dt / gamma
        if self._method == "trotter":
            self.info("  Trotter traction operator")
            self._trac_oper, self._trace = self._traction_operator(self._tf)
        elif self._method == "strang":
            self.info("  Strang traction operator")
            self._trac_oper, self._trace = self._traction_operator(self._tf / 2)

        self._std = np.sqrt(2 * self._kbt * self._tf)

        stability_tol = 1e-12
        max_eigenvalue = np.linalg.eigvals(self._trac_red).real.max()
        assert max_eigenvalue <= stability_tol, (
            f"Unstable mode detected ({max_eigenvalue:.3e}): increase lmax parameter"
        )
        self.info("System ready")

    @property
    def nsurf(self) -> int:
        """Return number of surface degrees of freedom."""
        return DIM * self.mesh[1]

    @property
    def nbulk(self) -> int:
        """Return number of bulk degrees of freedom."""
        return self.ndof - self.nsurf

    def _build_operators(self):
        """Build operators for the equations of motion."""
        inter = Interpolation(
            self._resolution, lmax=self._lmax, angular_grid=self._angular_grid
        )
        self._mesh = inter.mesh
        self._grid = inter.grid()

        ndof = self.ndof
        nsurf_pts = self.mesh[1]
        npoints = self.npoints
        nbulk_pts = npoints - nsurf_pts
        nbulk = self.nbulk
        eye = np.eye(DIM)

        # Derivative operators
        self.info("  derivative operator")
        transform = inter.transform()
        inverse = inter.inverse()
        partial_xyz = np.array(
            [inter.partial_x(), inter.partial_y(), inter.partial_z()]
        )

        # partial[p, a, n] = derivative in direction a at point p of basis coeff/vector n
        partial = np.einsum(
            "mM,aMN,Nn->man", inverse, partial_xyz, transform, optimize=True
        )

        # Traction operator
        self.info("  elastic traction operator")
        normals = self.normals()
        partial_surf = partial[-nsurf_pts:, :, :]  # (nsurf_pts, DIM, npoints)

        term0 = np.einsum("pa,pan->pn", normals, partial_surf)
        term1 = self._mu * np.einsum("pn,bc->pbnc", term0, eye)
        term2 = self._lam * np.einsum("pb,pcn->pbnc", normals, partial_surf)
        term3 = self._mu * np.einsum("pc,pbn->pbnc", normals, partial_surf)
        self._trac = (term1 + term2 + term3).reshape(DIM * nsurf_pts, DIM * npoints)

        # Stress-divergence operator
        self.info("  divergence of the elastic stress")

        # Reindex so P[a] is the full first-derivative matrix in direction a
        # P[a] shape: (npoints, npoints)
        P = np.moveaxis(partial, 1, 0)
        P_bulk = P[:, :nbulk_pts, :]  # shape: (DIM, nbulk_pts, npoints)

        gd = np.zeros((nbulk, ndof), dtype=float)

        # grad(div u): block (a,b) = d_a d_b
        for a in range(DIM):
            row = slice(a, None, DIM)
            Pa_bulk = P_bulk[a]
            for b in range(DIM):
                col = slice(b, None, DIM)
                gd[row, col] = Pa_bulk @ P[b]

        # vector Laplacian: same scalar Laplacian on each diagonal vector block
        lap = sum(P_bulk[c] @ P[c] for c in range(DIM))
        vl = np.zeros((nbulk, ndof), dtype=float)
        for a in range(DIM):
            row = slice(a, None, DIM)
            col = slice(a, None, DIM)
            vl[row, col] = lap

        self._ela = (self._mu + self._lam) * gd + self._mu * vl

        # Additive forces
        self.info("  additive forces")
        force = np.zeros((self.mesh[1], DIM))
        force[:, 2] = (self._ext) * np.array(
            [chi(np.array(self._loc)) for chi in inter._angular_functions]
        )
        self._force = force.ravel()
        self._build_surface_reduction()

    def _build_surface_reduction(self):
        """Precompute the linear elimination of the bulk degrees of freedom."""
        rank_tol = 1e-12
        cond_threshold = 1e12
        residual_threshold = 1e-8

        nsurf = self.nsurf
        bulk = slice(None, -nsurf)
        surf = slice(-nsurf, None)

        ebb = self._ela[:, bulk]
        ebs = self._ela[:, surf]
        expected_rank = min(ebb.shape)
        rank = np.linalg.matrix_rank(ebb, tol=rank_tol)
        condition = np.linalg.cond(ebb)

        self._bulk_from_surf = -np.linalg.lstsq(ebb, ebs, rcond=None)[0]
        residual = ebb @ self._bulk_from_surf + ebs
        residual_norm = np.linalg.norm(residual)

        self._surface_reduction_diagnostics = {
            "ebb_shape": ebb.shape,
            "expected_rank": int(expected_rank),
            "rank": int(rank),
            "condition_number": float(condition),
            "condition_threshold": float(cond_threshold),
            "residual_norm": float(residual_norm),
            "residual_threshold": float(residual_threshold),
        }

        if self.verbose:
            self.info(
                "  surface reduction diagnostics: "
                f"rank {rank}/{expected_rank}, cond {condition:.3e}, "
                f"residual {residual_norm:.3e}"
            )

        if (
            rank < expected_rank
            or not np.isfinite(condition)
            or condition > cond_threshold
            or residual_norm > residual_threshold
        ):
            raise ValueError(
                "Failed to eliminate bulk DOFs: ill-conditioned or inconsistent "
                "surface reduction system. Try adjusting mesh resolution and/or "
                "lmax."
            )
        self._trac_red = (
            self._trac[:, surf] + self._trac[:, bulk] @ self._bulk_from_surf
        )
        if self._init is not None:
            rigid = self._surface_rigid_modes(self._init)
            u, singular_values, _ = np.linalg.svd(rigid, full_matrices=False)
            if singular_values.size:
                rank = np.count_nonzero(singular_values > 1e-12 * singular_values[0])
                qrigid = u[:, :rank]
                project_rigid = qrigid @ qrigid.T
                project_elastic = np.eye(nsurf) - project_rigid
                self._trac_red = project_elastic @ self._trac_red @ project_elastic

    def _surface_rigid_modes(
        self, mode: Init = "no_rigid_motion"
    ) -> NDArray[np.floating]:
        """Return translational and rotational rigid modes on the surface."""
        angles = self.surface_points()[:, 1:]
        theta, phi = angles.T
        st = np.sin(theta)
        positions = np.stack(
            [st * np.cos(phi), st * np.sin(phi), np.cos(theta)],
            axis=1,
        )

        include_translations = mode in {"no_rigid_motion", "no_translations"}
        include_rotations = mode in {"no_rigid_motion", "no_rotations"}
        nmodes = DIM * int(include_translations) + DIM * int(include_rotations)
        if nmodes == 0:
            return np.zeros((self.nsurf, 0), dtype=float)

        modes = np.zeros((self.mesh[1], DIM, nmodes), dtype=float)
        idx = 0
        for axis in range(DIM):
            if include_translations:
                modes[:, axis, idx] = 1.0
                idx += 1
            if include_rotations:
                omega = np.eye(DIM)[axis]
                modes[:, :, idx] = np.cross(omega, positions)
                idx += 1
        return modes.reshape(self.nsurf, nmodes)

    def _reconstruct_state(self, surf: np.ndarray) -> VectorValues:
        """Reconstruct the full state from the surface degrees of freedom."""
        surf = np.asarray(surf).reshape(-1)
        vec = np.empty(self.ndof, dtype=surf.dtype)
        if self.nbulk:
            vec[: -self.nsurf] = self._bulk_from_surf @ surf
        vec[-self.nsurf :] = surf
        return vec.reshape(self.npoints, DIM)

    def _surface_state(self) -> np.ndarray:
        """Return flat surface degrees of freedom."""
        return self.surface().ravel()

    def info(self, message: str):
        """Log information message in the verbose mode."""
        if self.verbose:
            logger.info(message)

    def normals(self) -> VectorValues:
        """Return inward surface area normals."""
        angles = self._grid[-self.mesh[-1] :, 1:]
        st = np.sin(angles[:, 0])
        st2 = st**2
        ct = np.cos(angles[:, 0])
        sp = np.sin(angles[:, 1])
        cp = np.cos(angles[:, 1])
        return -np.stack([st2 * cp, st2 * sp, st * ct], axis=1)

    @property
    def resolution(self) -> tuple[int, int]:
        """Return resolution defined by the user."""
        return self._resolution

    @property
    def mesh(self) -> tuple[int, int]:
        """Return mesh specification of the point grid."""
        return self._mesh

    @property
    def lmax(self) -> int | None:
        """Return order of spherical-harmonic expansion of the Sobolev-Matérn kernels."""
        return self._lmax

    @property
    def npoints(self) -> int:
        """Return total number of the grid points."""
        return np.prod(self.mesh).item()

    @property
    def ndof(self) -> int:
        """Return total number of degrees of freedom."""
        return DIM * self.npoints

    def grid(self) -> Grid:
        """Return underlying mesh grid."""
        return self._grid

    def state(self) -> VectorValues:
        """Return copy of the state."""
        return np.array(self._state)

    def surface(self) -> VectorValues:
        """Return displacements of the surface points."""
        return np.array(self._state[-self.mesh[1] :])

    def surface_points(self) -> Grid:
        """Return surface points of the grid."""
        return np.array(self.grid()[-self.mesh[1] :])

    def __call__(self) -> Self:
        """Integration step.

        Returns:
            Updated system after one integration step.
        """
        if self._method == "euler":
            self._state = self._euler_step()
        elif self._method == "strang":
            self._state = self._strang_step()
        elif self._method == "trotter":
            self._state = self._trotter_step()

        return self

    def _solve_bulk(self, vec: np.ndarray) -> np.ndarray:
        """Compute displacements of the bulk degrees of freedom from the current surface degrees of freedom.

        Args:
            vec: Flat vector of displacements for all the degrees of freedom

        Returns:
            All updated degrees of freedom.
        """
        nsurf = self.nsurf
        bulk = slice(None, -nsurf)  # bulk selector
        surf = slice(-nsurf, None)  # surface selector

        vec[bulk] = self._bulk_from_surf @ vec[surf]
        return vec

    def _euler_step(self) -> VectorValues:
        """Perform one Euler-Maruyama integration step.

        Returns:
            Updated state vector after one Euler-Maruyama step.
        """
        surf = self._surface_state()

        # Surface step
        traction = self._trac_red @ surf
        flu = self._std * self._rng.normal(size=self.nsurf)
        surf = surf + (traction + self._force) * self._tf + flu
        return self._reconstruct_state(surf)

    def _traction_operator(self, tf: Real) -> tuple[LinearOperator, Real]:
        """Construct the traction operator for the exponential integrators.

        Args:
            tf: Time factor for the operator.

        Returns:
            The traction operator and its trace.
        """
        nsurf = self.nsurf
        trac_exp = self._trac_red * tf

        def matvec(x):
            x = np.asarray(x).reshape(-1)
            return trac_exp @ x

        def rmatvec(x):
            x = np.asarray(x).reshape(-1)
            return trac_exp.T @ x

        def matmat(x):
            x = np.asarray(x)
            return trac_exp @ x

        trace = np.trace(trac_exp)
        return cast(Any, LinearOperator)(
            shape=(nsurf, nsurf),
            matvec=matvec,
            rmatvec=rmatvec,
            matmat=matmat,
            dtype=float,
        ), trace

    def _trotter_step(self) -> VectorValues:
        """Perform one Trotter integration step.

        Returns:
            Updated state vector after one Trotter splitting step.
        """
        surf = self._surface_state()
        surf = expm_multiply(self._trac_oper, surf, traceA=self._trace)
        flu = self._std * self._rng.normal(size=self.nsurf)
        surf = surf + flu + self._force * self._tf
        return self._reconstruct_state(surf)

    def _strang_step(self) -> VectorValues:
        """Perform one Strang integration step.

        Returns:
            Updated state vector after one Strang splitting step.
        """
        surf = self._surface_state()
        surf = expm_multiply(self._trac_oper, surf, traceA=self._trace)
        flu = self._std * self._rng.normal(size=self.nsurf)
        surf = surf + flu + self._force * self._tf
        surf = expm_multiply(self._trac_oper, surf, traceA=self._trace)
        return self._reconstruct_state(surf)

    def save(self, filepath: str | Path) -> Self:
        """Save a checkpoint into an HDF5 file.

        Args:
            filepath: Path to the file to save data into.

        Raises:
            TypeError: On missing/invalid checkpoint fields or unknown RNG class.

        Returns:
            Self for fluent chaining.
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
            # Datasets
            f.create_dataset("points", data=self._grid, compression="lzf", shuffle=True)
            f.create_dataset("state", data=self._state, compression="lzf", shuffle=True)

            # Metadata as attributes
            f.attrs["resolution"] = self._resolution
            f.attrs["gamma"] = float(self._gamma)
            f.attrs["lambda"] = float(self._lam)
            f.attrs["mu"] = float(self._mu)
            f.attrs["dt"] = float(self._dt)
            f.attrs["f_ext"] = float(self._ext)
            f.attrs["f_loc"] = tuple(float(angle) for angle in self._loc)
            f.attrs["lmax"] = h5.optional_nnint(self._lmax)
            f.attrs["init"] = "none" if self._init is None else str(self._init)
            f.attrs["kT"] = float(self._kbt)
            f.attrs["method"] = str(self._method)
            f.attrs["angular_grid"] = str(self._angular_grid)
            f.attrs["rng_bit_generator"] = type(self._rng.bit_generator).__name__
            f.attrs["rng_state_json"] = json.dumps(
                self._rng.bit_generator.state, default=json_default
            )

        return self

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        verbose: bool = False,
        cache: str | Path | None = None,
    ) -> Self:
        """Load checkpoint data from an HDF5 file.

        Args:
            filepath: Path to the file to load the data from.
            verbose: Log the initialization.
            cache: Path to the cached mesh-associated data.

        Returns:
            Loaded instance of the system.

        Raises:
            ValueError: On missing/invalid checkpoint fields or unknown RNG class.
        """

        def decode_text(x: Any) -> str:
            if isinstance(x, (bytes, bytearray, np.bytes_)):
                return bytes(x).decode("utf-8")
            if isinstance(x, str):
                return x
            raise ValueError(f"Expected text attribute, got {type(x).__name__}")

        def restore_rng(obj: Any) -> Any:
            if isinstance(obj, list):
                return np.asarray(obj)
            if isinstance(obj, dict):
                return {k: restore_rng(v) for k, v in obj.items()}
            return obj

        with h5py.File(filepath, "r") as f:
            resolution = h5.attr_int_pair(f, "resolution")
            gamma = h5.attr_float(f, "gamma")
            lam = h5.attr_float(f, "lambda")
            mu = h5.attr_float(f, "mu")
            dt = h5.attr_float(f, "dt")
            f_ext = h5.attr_float(f, "f_ext")
            f_loc = h5.attr_float_pair(f, "f_loc")
            lmax = h5.attr_optional_nnint(f, "lmax")
            init = cast_init(h5.attr_any(f, "init"))
            kbt = h5.attr_float(f, "kT")
            method = cast(IntegrationMethod, decode_text(h5.attr_any(f, "method")))
            angular_grid = cast(
                AndularGrid, decode_text(h5.attr_any(f, "angular_grid"))
            )

            dset = cast(h5py.Dataset, f["state"])
            state = cast(NDArray[np.floating], dset[()])

            bitgen_name = decode_text(h5.attr_any(f, "rng_bit_generator"))
            state_json = decode_text(h5.attr_any(f, "rng_state_json"))
            rng_state = restore_rng(json.loads(state_json))

            bitgen_cls = getattr(np.random, bitgen_name, None)
            if bitgen_cls is None:
                raise ValueError(f"Unknown NumPy bit generator: {bitgen_name!r}")

            bitgen = bitgen_cls()
            cast(Any, bitgen).state = rng_state
            rng = np.random.Generator(bitgen)

            return cls(
                resolution=resolution,
                lam=lam,
                mu=mu,
                dt=dt,
                f_ext=f_ext,
                f_loc=f_loc,
                kbt=kbt,
                gamma=gamma,
                state=state,
                method=method,
                angular_grid=angular_grid,
                rng=rng,
                verbose=verbose,
                cache=cache,
                lmax=lmax,
                init=init,
            )

    def save_cache(self, filepath: str | Path) -> Self:
        """Save operator data in a cache file.

        Args:
            filepath: Path to the cache file.

        Returns:
            Self for fluent chaining.
        """
        with h5py.File(filepath, "w") as f:
            # Datasets
            f.create_dataset("grid", data=self._grid, compression="lzf", shuffle=True)
            f.create_dataset("trac", data=self._trac, compression="lzf", shuffle=True)
            f.create_dataset("ela", data=self._ela, compression="lzf", shuffle=True)
            f.create_dataset(
                "bulk_from_surf",
                data=self._bulk_from_surf,
                compression="lzf",
                shuffle=True,
            )
            f.create_dataset(
                "trac_red",
                data=self._trac_red,
                compression="lzf",
                shuffle=True,
            )
            f.create_dataset(
                "fext",
                data=self._force,
                compression="lzf",
                shuffle=True,
            )

            # Metadata as attributes
            f.attrs["resolution"] = self._resolution
            f.attrs["mesh"] = self.mesh
            f.attrs["lambda"] = float(self._lam)
            f.attrs["mu"] = float(self._mu)
            f.attrs["dt"] = float(self._dt)
            f.attrs["f_ext"] = float(self._ext)
            f.attrs["f_loc"] = tuple(float(angle) for angle in self._loc)
            f.attrs["lmax"] = h5.optional_nnint(self._lmax)
            f.attrs["angular_grid"] = str(self._angular_grid)
            f.attrs["init"] = "none" if self._init is None else str(self._init)

        return self

    def load_cache(self, filepath: str | Path) -> Self:
        """Load operator data from a cache file.

        Args:
            filepath: Path to the cache file.

        Returns:
            Self for fluent chaining.

        Raises:
            AssertionError: On inconsistent attributes between the system and the cache file.
            ValueError: On missing cache data in the file.
        """
        with h5py.File(filepath, "r") as f:
            resolution = h5.attr_int_pair(f, "resolution")
            lam = h5.attr_float(f, "lambda")
            mu = h5.attr_float(f, "mu")
            dt = h5.attr_float(f, "dt")
            f_ext = h5.attr_float(f, "f_ext")
            f_loc = h5.attr_float_pair(f, "f_loc")
            lmax = h5.attr_optional_nnint(f, "lmax")
            angular_grid = h5.attr_any(f, "angular_grid")
            init = cast_init(h5.attr_any(f, "init"))

            assert resolution == self._resolution, (
                "Inconsistent resolutions between the system and the cache file"
            )
            assert angular_grid == self._angular_grid, (
                "Inconsistent angular grid type between the system and the cache file"
            )
            assert lam == self._lam, (
                "Inconsistent Lamè coefficient lambda between the system and the cache file"
            )
            assert mu == self._mu, (
                "Inconsistent Lamè coefficient mu between the system and the cache file"
            )
            assert dt == self._dt, (
                "Inconsistent time steps between the system and the cache file"
            )
            assert f_ext == self._ext, (
                "Inconsistent force magnitudes between the system and the cache file"
            )
            assert f_loc == self._loc, (
                "Inconsistent force-application points between the system and the cache file"
            )
            assert lmax == self._lmax, (
                "Inconsistent spherical-harmonic cutoffs between the system and the cache file"
            )
            assert init == self._init, (
                "Inconsistent init projection modes between the system and the cache file"
            )

            # Datasets
            self._mesh = h5.attr_int_pair(f, "mesh")
            self._grid = h5.dataset_ndfloat(f, "grid")
            self._trac = h5.dataset_ndfloat(f, "trac")
            self._ela = h5.dataset_ndfloat(f, "ela")
            self._bulk_from_surf = h5.dataset_ndfloat(f, "bulk_from_surf")
            self._trac_red = h5.dataset_ndfloat(f, "trac_red")
            force = h5.dataset_ndfloat(f, "fext")
            if force.shape != (self.nsurf,):
                raise ValueError(
                    f"Expected cached force shape {(self.nsurf,)}, got {force.shape}"
                )
            self._force = force

        return self

    def _build_vts_projection(self, resolution: int) -> NDArray[np.floating]:
        """Return the surface-to-VTS projection matrix.

        Args:
            resolution (int): VTS resolution.

        Return:
            Constructed projection matrix.

        Raises:
            ValueError: One invalid resolution value.
        """
        if resolution < 2:
            raise ValueError("VTS export resolution must be at least 2")

        theta = np.linspace(0.0, np.pi, resolution)
        phi = np.linspace(0.0, 2.0 * np.pi, resolution)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        dense_angles = np.column_stack((theta_grid.ravel(), phi_grid.ravel()))

        inter = Interpolation(
            self.resolution,
            lmax=self.lmax,
            angular_grid=self._angular_grid,
        )
        return cast(
            NDArray[np.floating],
            inter._matern.chi(dense_angles).T * np.sqrt(inter._aws),
        )

    def save_projection_cache(
        self, filepath: str | Path, resolution: int = 100
    ) -> Self:
        """Save VTS projection data in a cache file.

        Args:
            filepath: Path to the projection cache file.
            resolution: Number of dense points in each VTS angular direction.

        Returns:
            Self for fluent chaining.
        """
        projection = self._build_vts_projection(resolution)

        with h5py.File(filepath, "w") as f:
            f.create_dataset(
                "projection",
                data=projection,
                compression="lzf",
                shuffle=True,
            )

            f.attrs["resolution"] = self._resolution
            f.attrs["mesh"] = self.mesh
            f.attrs["vts_resolution"] = int(resolution)
            f.attrs["lmax"] = h5.optional_nnint(self._lmax)
            f.attrs["angular_grid"] = str(self._angular_grid)

        return self

    def load_projection_cache(
        self, filepath: str | Path, resolution: int | None = None
    ) -> NDArray[np.floating]:
        """Load VTS projection data from a cache file.

        Args:
            filepath: Path to the projection cache file.
            resolution: Expected number of dense points in each VTS angular direction.

        Returns:
            Cached VTS projection matrix.

        Raises:
            AssertionError: On inconsistent attributes between the system and the cache file.
            ValueError: On missing or malformed projection data in the file.
        """
        with h5py.File(filepath, "r") as f:
            cache_resolution = h5.attr_int_pair(f, "resolution")
            cache_mesh = h5.attr_int_pair(f, "mesh")
            cache_vts_resolution = h5.attr_int(f, "vts_resolution")
            cache_lmax = h5.attr_optional_nnint(f, "lmax")
            cache_angular_grid = h5.attr_any(f, "angular_grid")

            assert cache_resolution == self._resolution, (
                "Inconsistent resolutions between the system and the projection cache file"
            )
            assert cache_mesh == self.mesh, (
                "Inconsistent meshes between the system and the projection cache file"
            )
            assert cache_lmax == self._lmax, (
                "Inconsistent spherical-harmonic cutoffs between the system and the projection cache file"
            )
            assert cache_angular_grid == self._angular_grid, (
                "Inconsistent angular grid type between the system and the projection cache file"
            )
            if resolution is not None:
                assert cache_vts_resolution == resolution, (
                    "Inconsistent VTS export resolutions between the system and the projection cache file"
                )

            if "projection" not in f:
                raise ValueError("Missing cached projection data")
            projection_obj = f["projection"]
            if not isinstance(projection_obj, h5py.Dataset):
                raise ValueError("Cached projection data must be an HDF5 dataset")
            projection = np.array(projection_obj[()], dtype=float)

        expected_shape = (
            cache_vts_resolution * cache_vts_resolution,
            self.mesh[1],
        )
        if projection.shape != expected_shape:
            raise ValueError(
                f"Expected cached projection shape {expected_shape}, got {projection.shape}"
            )
        return cast(NDArray[np.floating], projection)

    @staticmethod
    def _spherical_to_cartesian(points: Grid) -> VectorValues:
        """Convert spherical `(r, theta, phi)` points to Cartesian points.

        Args:
            points: Spherical coordinates of 3D points.

        Returns:
            Cartesian coordinates of the 3D points.
        """
        radius, theta, phi = np.asarray(points).T
        return np.column_stack(
            (
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            )
        )

    @staticmethod
    def _export_path(path: str | Path, suffix: str) -> Path:
        """Create path and check the suffix of VTK file to export.

        Args:
            path: Name of the file to export.
            suffix: Suffix (".vts" or ".vtp") determines the exported format.

        Returns:
            Path with the proper suffix.

        Raises:
            ValueError: On inconsistent suffix.
        """
        output = Path(path)
        if output.suffix:
            if output.suffix != suffix:
                raise ValueError(
                    f"Expected output suffix {suffix!r}, got {output.suffix!r}"
                )
        else:
            output = output / f"surface{suffix}"
        output.parent.mkdir(parents=True, exist_ok=True)
        return output

    @staticmethod
    def _vtk_data(values: np.ndarray) -> str:
        """Format array values for an ASCII VTK XML data array.

        Args:
            values: Array to format.

        Returns:
            Formatted array.
        """
        flat = np.asarray(values).ravel()
        if np.issubdtype(flat.dtype, np.integer):
            return " ".join(str(int(value)) for value in flat)
        return " ".join(f"{float(value):.17g}" for value in flat)

    @staticmethod
    def _vtk_data_array(
        name: str | None,
        values: np.ndarray,
        number_of_components: int | None = None,
        dtype: str = "Float64",
    ) -> str:
        """Return one ASCII VTK XML ``DataArray`` block."""
        attrs = [f'type="{dtype}"']
        if name is not None:
            attrs.append(f'Name="{name}"')
        if number_of_components is not None:
            attrs.append(f'NumberOfComponents="{number_of_components}"')
        attrs.append('format="ascii"')
        return (
            f"        <DataArray {' '.join(attrs)}>"
            f"{System._vtk_data(values)}</DataArray>\n"
        )

    @staticmethod
    def _orient_faces(
        faces: NDArray[np.integer], points: VectorValues
    ) -> NDArray[np.integer]:
        """Orient surface polygons outward with respect to reference points."""
        oriented = np.array(faces, dtype=np.int64, copy=True)
        for i, face in enumerate(oriented):
            polygon = points[face]
            normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
            centroid = np.mean(polygon, axis=0)
            if np.dot(normal, centroid) < 0:
                oriented[i] = face[::-1]
        return oriented

    def _surface_faces(self) -> NDArray[np.integer]:
        """Return native surface polygon connectivity."""
        points = self._spherical_to_cartesian(self.surface_points())
        if self._angular_grid == "fibonacci":
            hull = ConvexHull(points)
            return self._orient_faces(hull.simplices, points)

        if self._angular_grid == "gauss-legendre-fourier":
            n_latitude = self.resolution[1]
            n_longitude = self.mesh[1] // n_latitude
            faces = []
            for lat in range(n_latitude - 1):
                row = lat * n_longitude
                next_row = (lat + 1) * n_longitude
                for lon in range(n_longitude):
                    next_lon = (lon + 1) % n_longitude
                    faces.append(
                        [
                            row + lon,
                            row + next_lon,
                            next_row + next_lon,
                            next_row + lon,
                        ]
                    )
            return self._orient_faces(np.array(faces, dtype=np.int64), points)

        raise ValueError(f"Unknown angular grid: {self._angular_grid!r}")

    def export_vts(
        self,
        path: str | Path,
        resolution: int = 100,
        projection_cache: str | Path | None = None,
    ) -> Self:
        """Interpolate the current surface and export it as a VTK StructuredGrid.

        Args:
            path: Output file name.
            resolution: Number of dense points in each angular direction.
            projection_cache: Optional path to a cached VTS projection matrix.

        Returns:
            Self for fluent chaining.
        """
        if resolution < 2:
            raise ValueError("VTS export resolution must be at least 2")

        output = self._export_path(path, ".vts")
        theta = np.linspace(0.0, np.pi, resolution)
        phi = np.linspace(0.0, 2.0 * np.pi, resolution)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        dense_angles = np.column_stack((theta_grid.ravel(), phi_grid.ravel()))

        radius = float(np.mean(self.surface_points()[:, 0]))
        reference_points = self._spherical_to_cartesian(
            np.column_stack((np.full(len(dense_angles), radius), dense_angles))
        )

        projection = (
            self._build_vts_projection(resolution)
            if projection_cache is None
            else self.load_projection_cache(projection_cache, resolution=resolution)
        )
        displacement = projection @ self.surface()
        points = reference_points + displacement
        magnitude = np.linalg.norm(displacement, axis=1)
        extent = f"0 {resolution - 1} 0 {resolution - 1} 0 0"

        content = (
            '<?xml version="1.0"?>\n'
            '<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n'
            f'  <StructuredGrid WholeExtent="{extent}">\n'
            f'    <Piece Extent="{extent}">\n'
            '      <PointData Vectors="Displacement_Vector" '
            'Scalars="Displacement_Magnitude">\n'
            + self._vtk_data_array(
                "Displacement_Vector",
                displacement,
                number_of_components=DIM,
            )
            + self._vtk_data_array("Displacement_Magnitude", magnitude)
            + "      </PointData>\n"
            + "      <Points>\n"
            + self._vtk_data_array(None, points, number_of_components=DIM)
            + "      </Points>\n"
            + "    </Piece>\n"
            + "  </StructuredGrid>\n"
            + "</VTKFile>\n"
        )
        output.write_text(content)
        return self

    def export_vtp(self, path: str | Path) -> Self:
        """Export the native current surface as VTK PolyData.

        Args:
            path: Output file.

        Returns:
            Self for fluent chaining.
        """
        output = self._export_path(path, ".vtp")
        displacement = self.surface()
        points = self._spherical_to_cartesian(self.surface_points()) + displacement
        magnitude = np.linalg.norm(displacement, axis=1)
        point_ids = np.arange(len(points), dtype=np.int64)
        faces = self._surface_faces()
        connectivity = faces.ravel()
        offsets = np.cumsum(np.full(len(faces), faces.shape[1], dtype=np.int64))

        content = (
            '<?xml version="1.0"?>\n'
            '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
            "  <PolyData>\n"
            f'    <Piece NumberOfPoints="{len(points)}" '
            f'NumberOfPolys="{len(faces)}">\n'
            '      <PointData Vectors="Displacement_Vector" '
            'Scalars="Displacement_Magnitude">\n'
            + self._vtk_data_array(
                "Displacement_Vector",
                displacement,
                number_of_components=DIM,
            )
            + self._vtk_data_array("Displacement_Magnitude", magnitude)
            + self._vtk_data_array("Surface_Point_Id", point_ids, dtype="Int64")
            + "      </PointData>\n"
            + "      <Points>\n"
            + self._vtk_data_array(None, points, number_of_components=DIM)
            + "      </Points>\n"
            + "      <Polys>\n"
            + self._vtk_data_array("connectivity", connectivity, dtype="Int64")
            + self._vtk_data_array("offsets", offsets, dtype="Int64")
            + "      </Polys>\n"
            + "    </Piece>\n"
            + "  </PolyData>\n"
            + "</VTKFile>\n"
        )
        output.write_text(content)
        return self
