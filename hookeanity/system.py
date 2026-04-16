"""Spherical body subject to fluctuating forces and point-source surface forces.

Attributes:
    Seed (typing.TypeAlias): Random-number generator seed
"""

from typing import cast, Any, Self, TypeAlias
import numpy as np
from scipy.sparse.linalg import expm_multiply, LinearOperator
from numpy.typing import NDArray
from .interpolation import Interpolation
from .base import DIM, Real, Grid, VectorValues, instance
import h5py
import json
from pathlib import Path
from .logging import logger

Seed: TypeAlias = int | np.integer


def attr_any(file: h5py.File, key: str) -> Any:
    """Read an attribute from an HDF5 file handle.

    Args:
        file: File handle
        key: Attribute name

    Returns:
        Attribute value
    """
    return cast(Any, file.attrs[key])


def attr_float(file: h5py.File, key: str) -> float:
    """Read floating-point attribute from an HDF5 file handle.

    Args:
        file: File handle
        key: Attribute name

    Returns:
        Attribute value
    """
    return float(attr_any(file, key))


def attr_mesh(file: h5py.File) -> tuple[int, int]:
    """Read mesh scheme from an HDF5 file handle attributes.

    Args:
        file: File handle

    Returns:
        Mesh specification

    Raises:
        ValueError When mesh attribute is malformed
    """
    key = "mesh"
    arr = np.asarray(attr_any(file, key))
    if arr.shape != (2,):
        raise ValueError(f"Expected {key} shape (2,), got {arr.shape}")
    return (int(arr[0]), int(arr[1]))


def attr_f_loc(file: h5py.File) -> tuple[float, float]:
    """Read point of the external force application from an HDF5 file handle attributes.

    Args:
        file: File handle

    Returns:
        Location of the force application point

    Raises:
        ValueError When f_loc attribute is malformed
    """
    key = "f_loc"
    arr = np.asarray(attr_any(file, key))
    if arr.shape != (2,):
        raise ValueError(f"Expected {key} shape (2,), got {arr.shape}")
    return (float(arr[0]), float(arr[1]))


def ndfloat_dataset(file: h5py.File, key: str) -> NDArray[np.floating]:
    """Load numpy array of floating-point numbers from an HDF5 file handle.

    Args:
        file: File handle
        key: Dataset key

    Returns:
        Numpy error loaded from the file
    """
    dset = cast(h5py.Dataset, file[key])
    return np.array(dset[()], dtype=float)


class System:
    """Simulated system.

    Attributes:
        _gamma (hookeanity.Real): Effective dissipative coefficient
        _lam (hookeanity.Real): Lamé lambda coefficient
        _mu (hookeanity.Real): Lamé mu coefficient
        _ext (hookeanity.Real): Magnitude of the external point force
        _loc (tuple[hookeanity.Real, hookeanity.Real]): Solid angle specifying the location of the external point force
        _kbt (hookeanity.Real): Temperature
        _state (hookeanity.VectorValues): Displacements of the grid points
        _grid (kookeanity.Grid): Mesh grid
        _inter (hookeanity.interpolation.Interpolation): Interpolation utilities
        _rng (numpy.random.Generator): Numpy random-number generator
        _trac (numpy.ndarray): Matrix operator of the elastic traction force
        _trac_oper (numpy.ndarray): Matrix traction operator for Trotter or Strang methods of integration
        _trace (hookeanity.Real)" Trace of the matrix traction operator for Trotter or Strang methods of integration
        _ela (numpy.ndarray): Matrix operator of the divergence of the elastic stress
        _force (numpy.ndarray): External point force
        _std (numpy.ndarray): Intensity of the thermal fluctuations
        _method (str): Integration method ("euler", "trotter", "strang")
        _dt (hookeanity.Real): Integration time step
        _tf (hookeanity.Real): Time step normalized by the dissipative coefficient
        verbose (bool): Trigger of logging output
    """

    def __init__(
        self,
        mesh: tuple[int, int],
        lam: Real,
        mu: Real,
        dt: Real,
        f_ext: Real = 0.0,
        f_loc: tuple[Real, Real] = (0.0, 0.0),
        kbt: Real = 1.0,
        gamma: Real = 1.0,
        state: VectorValues | None = None,
        method: str = "euler",
        rng: np.random.Generator | Seed | None = None,
        verbose: bool = False,
        cache: str | Path | None = None,
    ):
        """Initialize the simulation system.

        Args:
            mesh: Number of radial and angular grid points
            lam: Lamé's first parameter
            mu: Lamé's second parameter (shear modulus)
            dt: Integration time step
            f_ext: Magnitude of the external force defaults to zero
            f_loc: Solid angle specifying the location of the external point force
            kbt: Effective temperature in units of energy defaults to one
            gamma: Effective dissipative coefficient
            state: Current state of the system defaults to zeros
            method: Integration method ("euler", "trotter", "strang")
            rng: Optional state of the random number generator or seed
            verbose: Log the initialization
            cache: Path to the cached numerical data

        Raises:
            ValueError: On unknown integration method
        """
        if method not in {"euler", "strang", "trotter"}:
            raise ValueError(f"Unknown integration method: {method!r}.")

        self._mesh = mesh
        self._gamma = gamma
        self._lam = lam
        self._mu = mu
        self._dt = dt
        self._ext = f_ext
        self._loc = f_loc
        self._kbt = kbt
        self._method = method
        self._state = (
            np.zeros((np.prod(mesh), DIM)) if state is None else np.array(state)
        )
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

        self._tf = dt / gamma
        if self._method == "trotter":
            self.info("  Trotter traction operator")
            self._trac_oper, self._trace = self._traction_operator(self._tf)
        elif self._method == "strang":
            self.info("  Strang traction operator")
            self._trac_oper, self._trace = self._traction_operator(self._tf / 2)

        self._std = np.sqrt(2 * self._kbt * self._tf)
        self.info("System ready")

    def _build_operators(self):
        """Build operators for the equations of motion."""
        mesh = self._mesh
        ndof = self.ndof
        nsurf_pts = mesh[1]
        npoints = self.npoints
        nbulk_pts = npoints - nsurf_pts
        nbulk = DIM * nbulk_pts
        eye = np.eye(DIM)

        inter = Interpolation(mesh)
        self._grid = inter.grid()

        # Derivative operators
        self.info("  derivative operator")
        transform = inter.transform()
        inverse = inter.inverse()
        partial_xyz = np.array(
            [inter.partial_x(), inter.partial_y(), inter.partial_z()]
        )

        # partial[p, a, n] = derivative in direction a at point p of basis coeff/vector n
        partial = np.einsum("mM,aMN,Nn->man", inverse, partial_xyz, transform)

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
        omega = 4 * np.pi / mesh[1] # Fibonacci weight
        force = np.zeros((mesh[1], DIM))
        force[:, 2] = (self._ext / np.sqrt(omega))* np.array(
            [chi(np.array(self._loc)) for chi in inter._angular_functions]
        )
        self._force = force.ravel()

    def info(self, message: str):
        """Log information message in the verbose mode."""
        if self.verbose:
            logger.info(message)

    def normals(self) -> VectorValues:
        """Return inward surface area normals dS."""
        angles = self._grid[-self.mesh[-1] :, 1:]
        st = np.sin(angles[:, 0])
        st2 = st**2
        ct = np.cos(angles[:, 0])
        sp = np.sin(angles[:, 1])
        cp = np.cos(angles[:, 1])
        return -np.stack([st2 * cp, st2 * sp, st * ct], axis=1)

    @property
    def mesh(self) -> tuple[int, int]:
        """Return mesh specification of the point grid."""
        return self._mesh

    @property
    def npoints(self) -> int:
        """Return number of the grid points."""
        return np.prod(self.mesh).item()

    @property
    def ndof(self) -> int:
        """Return number of degrees of freedm."""
        return DIM * self.npoints

    def grid(self) -> Grid:
        """Return generator of quadrature points."""
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
            Updated system after one integration step
        """
        if self._method == "euler":
            self._state = self._euler_step()
        elif self._method == "strang":
            self._state = self._strang_step()
        elif self._method == "trotter":
            self._state = self._trotter_step()

        return self

    def _solve_bulk(self, vec: np.ndarray) -> np.ndarray:
        """Solve the bulk equilibrium equations for the given surface state.

        Args:
            vec: Flat vector of displacements for all the degrees of freedom

        Returns:
            Updated degrees of freedom
        """
        nsurf = DIM * self.mesh[1]  # number of surface DoFs
        bulk = slice(None, -nsurf)  # bulk selector
        surf = slice(-nsurf, None)  # surface selector

        vec[bulk] = np.linalg.solve(self._ela[:, bulk], -self._ela[:, surf] @ vec[surf])
        return vec

    def _euler_step(self) -> VectorValues:
        """Perform one Euler-Maruyama integration step.

        Returns:
            Updated state vector after one Euler-Maruyama step.
        """
        nsurf = DIM * self.mesh[1]  # nuber of surface DoFs
        surf = slice(-nsurf, None)  # surface selector
        vec = self.state().ravel()  # State Vector

        # Surface step
        traction = self._trac @ vec
        flu = self._std * self._rng.normal(size=nsurf)
        vec[surf] += (traction + self._force) * self._tf + flu

        # Bulk step
        self._solve_bulk(vec)
        return vec.reshape(self.npoints, DIM)

    def _traction_operator(self, tf: Real) -> tuple[LinearOperator, Real]:
        """Construct the traction operator for the exponential integrators.

        Args:
            tf: Time factor for the operator

        Returns:
            The traction operator and its trace
        """
        ndof = self.ndof
        nsurf = DIM * self.mesh[1]  # nuber of surface DoFs
        surf = slice(-nsurf, None)  # surface selector
        trac_exp = self._trac * tf

        def matvec(x):
            x = np.asarray(x).reshape(-1)
            y = np.zeros(ndof, dtype=x.dtype)
            y[surf] = trac_exp @ x
            return y

        def rmatvec(x):
            x = np.asarray(x).reshape(-1)
            return trac_exp.T @ x[surf]

        def matmat(x):
            x = np.asarray(x)
            y = np.zeros((ndof, x.shape[1]), dtype=x.dtype)
            y[surf, :] = trac_exp @ x
            return y

        trace = np.trace(trac_exp[:, surf])
        return cast(Any, LinearOperator)(
            shape=(ndof, ndof),
            matvec=matvec,
            rmatvec=rmatvec,
            matmat=matmat,
            dtype=float,
        ), trace

    def _trotter_step(self) -> VectorValues:
        """Perform one Trotter splitting integration step.

        Returns:
            Updated state vector after one Trotter splitting step
        """
        nsurf = DIM * self.mesh[1]  # nuber of surface DoFs
        surf = slice(-nsurf, None)  # surface selector

        vec = self.state().ravel()  # State Vector
        vec = expm_multiply(self._trac_oper, vec, traceA=self._trace)

        flu = self._std * self._rng.normal(size=nsurf)
        vec[surf] += flu + self._force * self._tf
        self._solve_bulk(vec)

        return vec.reshape(self.npoints, DIM)

    def _strang_step(self) -> VectorValues:
        """Perform one Strang splitting integration step.

        Returns:
            Updated state vector after one Strang splitting step
        """
        nsurf = DIM * self.mesh[1]  # nuber of surface DoFs
        surf = slice(-nsurf, None)  # surface selector

        vec = self.state().ravel()  # State Vector
        vec = expm_multiply(self._trac_oper, vec, traceA=self._trace)

        flu = self._std * self._rng.normal(size=nsurf)
        vec[surf] += flu + self._force * self._tf
        self._solve_bulk(vec)

        vec = expm_multiply(self._trac_oper, vec, traceA=self._trace)
        self._solve_bulk(vec)
        return vec.reshape(self.npoints, DIM)

    def save(self, filepath: str | Path) -> Self:
        """Save system data into an HDF5 file.

        Args:
            filepath: Path to the file to save data into

        Raises:
            TypeError: On missing/invalid checkpoint fields or unknown RNG class

        Returns:
            Self for fluent chaining
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
            f.attrs["mesh"] = self.mesh
            f.attrs["gamma"] = float(self._gamma)
            f.attrs["lambda"] = float(self._lam)
            f.attrs["mu"] = float(self._mu)
            f.attrs["dt"] = float(self._dt)
            f.attrs["f_ext"] = float(self._ext)
            f.attrs["f_loc"] = tuple(float(angle) for angle in self._loc)
            f.attrs["kT"] = float(self._kbt)
            f.attrs["method"] = str(self._method)
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
        """Load system data from an HDF5 file.

        Args:
            filepath: Path to the file to load the data from
            verbose: Log the initialization
            cache: Path to the cached mesh-associated data

        Raises:
            ValueError: On missing/invalid checkpoint fields or unknown RNG class

        Returns:
            Loaded instance of the system
        """

        def decode_text(x: Any) -> str:
            if isinstance(x, (bytes, bytearray, np.bytes_)):
                return bytes(x).decode("utf-8")
            if isinstance(x, str):
                return x
            raise ValueError(f"Expected text attribute, got {type(x).__name__}")

        def restore_numpy(obj: Any) -> Any:
            if isinstance(obj, list):
                return np.asarray(obj)
            if isinstance(obj, dict):
                return {k: restore_numpy(v) for k, v in obj.items()}
            return obj

        with h5py.File(filepath, "r") as f:
            mesh = attr_mesh(f)

            gamma = attr_float(f, "gamma")
            lam = attr_float(f, "lambda")
            mu = attr_float(f, "mu")
            dt = attr_float(f, "dt")
            f_ext = attr_float(f, "f_ext")
            f_loc = attr_f_loc(f)
            kbt = attr_float(f, "kT")
            method = decode_text(attr_any(f, "method"))

            dset = cast(h5py.Dataset, f["state"])
            state = cast(NDArray[np.floating], dset[()])

            bitgen_name = decode_text(attr_any(f, "rng_bit_generator"))
            state_json = decode_text(attr_any(f, "rng_state_json"))
            rng_state = restore_numpy(json.loads(state_json))

            bitgen_cls = getattr(np.random, bitgen_name, None)
            if bitgen_cls is None:
                raise ValueError(f"Unknown NumPy bit generator: {bitgen_name!r}")

            bitgen = bitgen_cls()
            cast(Any, bitgen).state = rng_state
            rng = np.random.Generator(bitgen)

            return cls(
                mesh=mesh,
                lam=lam,
                mu=mu,
                dt=dt,
                f_ext=f_ext,
                f_loc=f_loc,
                kbt=kbt,
                gamma=gamma,
                state=state,
                method=method,
                rng=rng,
                verbose=verbose,
                cache=cache,
            )

    def save_cache(self, filepath: str | Path) -> Self:
        """Save mesh in a cache file.

        Args:
            filepath: Path to the cache file

        Returns:
            Self for fluent chaining
        """
        with h5py.File(filepath, "w") as f:
            # Datasets
            f.create_dataset("grid", data=self._grid, compression="lzf", shuffle=True)
            f.create_dataset("trac", data=self._trac, compression="lzf", shuffle=True)
            f.create_dataset("ela", data=self._ela, compression="lzf", shuffle=True)
            f.create_dataset(
                "fext",
                data=self._force[DIM - 2 :: DIM],
                compression="lzf",
                shuffle=True,
            )

            # Metadata as attributes
            f.attrs["mesh"] = self.mesh
            f.attrs["lambda"] = float(self._lam)
            f.attrs["mu"] = float(self._mu)
            f.attrs["f_ext"] = float(self._ext)
            f.attrs["f_loc"] = tuple(float(angle) for angle in self._loc)

        return self

    def load_cache(self, filepath: str | Path) -> Self:
        """Save mesh in a cache file.

        Args:
            filepath: Path to the cache file

        Returns:
            Self for fluent chaining

        Raises:
            AssertionError On inconsistent attributes between the system and the cache file
            ValueError On missing cache data in the file
        """
        with h5py.File(filepath, "r") as f:
            mesh = attr_mesh(f)
            lam = attr_float(f, "lambda")
            mu = attr_float(f, "mu")
            f_ext = attr_float(f, "f_ext")
            f_loc = attr_f_loc(f)

            assert mesh == self._mesh, (
                "Inconsistent meshes between the system and the cache file"
            )
            assert lam == self._lam, (
                "Inconsistent Lamè coefficient lambda between the system and the cache file"
            )
            assert mu == self._mu, (
                "Inconsistent Lamè coefficient mu between the system and the cache file"
            )
            assert f_ext == self._ext, (
                "Inconsistent force magnitudes between the system and the cache file"
            )
            assert f_loc == self._loc, (
                "Inconsistent force-application points between the system and the cache file"
            )

            # Datasets
            self._grid = ndfloat_dataset(f, "grid")
            self._trac = ndfloat_dataset(f, "trac")
            self._ela = ndfloat_dataset(f, "ela")
            force = np.zeros((mesh[1], DIM))
            force[:, 2] = ndfloat_dataset(f, "fext")
            self._force = force.ravel()

        return self
