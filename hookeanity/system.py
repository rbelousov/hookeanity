"""Spherical body subject to fluctuating forces and point-source surface forces.

Attributes:
    Seed (typing.TypeAlias): Random-number generator seed
"""

from typing import cast, Any, Self, TypeAlias
from collections.abc import Iterator
import numpy as np
from numpy.typing import NDArray
from .interpolation import Interpolation
from .base import DIM, Real, Grid, VectorValues, instance
import h5py
import json
from pathlib import Path

Seed: TypeAlias = int | np.integer


class System:
    """Simulated system.

    Attributes:
        _lam (hookeanity.Real): Lamé lambda coefficient
        _mu (hookeanity.Real): Lamé mu coefficient
        _ext (hookeanity.Real): Magnitude of the external point force
        _loc (tuple[hookeanity.Real, hookeanity.Real]): Solid angle specifying the location of the external point force
        _kbt (hookeanity.Real): Temperature
        _state (VectorValues): Displacements of the grid points
        _inter (hookeanity.interpolation.Interpolation): Interpolation utilities
        _rng (numpy.random.Generator): Numpy random-number generator
        _trac (numpy.ndarray): Matrix operator of the elastic traction force
        _ela_surf (numpy.ndarray): Matrix operator of the divergence of the elastic stress on the surface
        _ela_pinv (numpy.ndarray): Pseudoinverse matrix operator of the divergence of the elastic stress in the bulk
        _force (numpy.ndarray): External point force
        _std (numpy.ndarray): Intensity of the thermal fluctuations
    """

    def __init__(
        self,
        mesh: tuple[int, int],
        lam: Real,
        mu: Real,
        f_ext: Real = 0.0,
        f_loc: tuple[Real, Real] = (0.0, 0.0),
        kbt: Real = 1.0,
        state: VectorValues | None = None,
        rng: np.random.Generator | Seed | None = None,
    ):
        """Initialize the simulation system.

        Args:
            mesh: Number of radial and angular grid points
            lam: Lamé's first parameter
            mu: Lamé's second parameter (shear modulus)
            f_ext: Magnitude of the external force defaults to zero
            f_loc: Solid angle specifying the location of the external point force
            kbt: Effective temperature in units of energy defaults to one
            state: Current state of the system defaults to zeros
            rng: Optional state of the random number generator or seed
        """
        self._lam = lam
        self._mu = mu
        self._ext = f_ext
        self._loc = f_loc
        self._kbt = kbt
        self._inter = Interpolation(mesh)
        self._state = (
            np.zeros((np.prod(mesh), DIM)) if state is None else np.array(state)
        )
        self._rng = (
            rng
            if instance(rng, np.random.Generator)
            else np.random.default_rng(rng if instance(rng, Seed) else None)
        )
        # self._vec = self._state.ravel()

        # Derivative operator
        transform = self._inter.transform()
        inverse = self._inter.inverse()
        partial_xyz = np.array(
            [self._inter.partial_x(), self._inter.partial_y(), self._inter.partial_z()]
        )
        partial = np.einsum("mM,aMN,Nn->man", inverse, partial_xyz, transform)

        eye = np.eye(DIM)
        ndof = self.ndof
        nsurf = DIM * mesh[1]
        nbulk = ndof - nsurf
        npoints = self.npoints
        bulk = slice(None, -nsurf)  # bulk selector
        surf = slice(-nsurf, None)  # surface selector

        # Traction operator
        normals = self.normals()
        prj = np.kron(
            np.eye(npoints) - np.ones((npoints, npoints)) / npoints, eye
        )  # CoM removal
        term0 = np.einsum("pa,pan->pn", normals, partial[-mesh[1] :, :, :])
        term1 = self._mu * np.einsum("pn,bc->pbnc", term0, eye)
        term2 = self._lam * np.einsum(
            "pb,pcn->pbnc", normals, partial[-mesh[1] :, :, :]
        )
        term3 = self._mu * np.einsum("pc,pbn->pbnc", normals, partial[-mesh[1] :, :, :])
        trac = (term1 + term2 + term3).reshape(DIM * mesh[1], DIM * np.prod(mesh))
        self._trac = trac @ prj

        gd = np.einsum(
            "lL,aLM,bMN,Nn->lanb",
            inverse[: -mesh[1]],
            partial_xyz,
            partial_xyz,
            transform,
        ).reshape(nbulk, ndof)  # grad div
        lap = np.einsum(
            "ab,lL,cLM,cMN,Nn->lanb",
            np.eye(DIM),
            inverse[: -mesh[1]],
            partial_xyz,
            partial_xyz,
            transform,
        ).reshape(nbulk, ndof)  # laplacian
        ela = (self._mu + self._lam) * gd + self._mu * lap  # div stress
        self._ela_surf = ela[:, surf]
        self._ela_pinv = np.linalg.pinv(ela[:, bulk])

        force = np.zeros((mesh[1], DIM))
        force[:, 2] = self._ext * np.array(
            [chi(np.array(self._loc)) for chi in self._inter._angular_functions]
        )
        self._force = force.ravel()
        self._std = np.sqrt(2 * self._kbt)

    def normals(self) -> VectorValues:
        """Return inward surface normals."""
        st = np.sin(self._inter._angles[:, 0])
        st2 = st**2
        ct = np.cos(self._inter._angles[:, 0])
        sp = np.sin(self._inter._angles[:, 1])
        cp = np.cos(self._inter._angles[:, 1])
        return -np.stack([st2 * cp, st2 * sp, st * ct], axis=1)

    @property
    def mesh(self) -> tuple[int, int]:
        """Return mesh specification of the point grid."""
        return self._inter.mesh

    @property
    def npoints(self) -> int:
        """Return number of the grid points."""
        return np.prod(self.mesh).item()

    @property
    def ndof(self) -> int:
        """Return number of degrees of freedm."""
        return DIM * self.npoints

    def points(self) -> Iterator[NDArray[np.floating]]:
        """Return generator of quadrature points."""
        return self._inter.points()

    def grid(self) -> Grid:
        """Return generator of quadrature points."""
        return self._inter.grid()

    def state(self) -> VectorValues:
        """Return copy of the state."""
        return np.array(self._state)

    def surface(self) -> VectorValues:
        """Return displacements of the surface points."""
        return np.array(self._state[-self.mesh[1] :])

    def surface_points(self) -> Grid:
        """Return surface points of the grid."""
        return np.array(self.grid()[-self.mesh[1] :])

    def __call__(self, dt: Real) -> Self:
        """Euler-Maruyama integration step.

        Args:
            dt: Integration time step

        Returns:
            Updated system after one integration step.
        """
        mesh = self.mesh
        nsurf = DIM * mesh[1]  # nuber of surface DoFs
        bulk = slice(None, -nsurf)  # bulk selector
        surf = slice(-nsurf, None)  # surface selector
        vec = self.state().ravel()  # State Vector

        # Surface step
        traction = self._trac @ vec
        flu = (np.sqrt(dt) * self._std) * self._rng.normal(size=nsurf)
        vec[surf] += dt * (traction + self._force) + flu

        # Bulk step
        # vec[bulk] = np.linalg.solve(self._ela[:, bulk], -self._ela[:, surf] @ vec[surf])
        vec[bulk] = -self._ela_pinv @ self._ela_surf @ vec[surf]

        self._state = vec.reshape(self.npoints, DIM)
        return self

    def save(self, filepath: str | Path) -> Self:
        """Save system data into an HDF5 file."""

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
            f.create_dataset("points", data=self.grid())
            f.create_dataset("state", data=self._state, compression="lzf", shuffle=True)

            # Metadata as attributes
            f.attrs["mesh"] = self.mesh
            f.attrs["lambda"] = float(self._lam)
            f.attrs["mu"] = float(self._mu)
            f.attrs["f_ext"] = float(self._ext)
            f.attrs["f_loc"] = tuple(float(angle) for angle in self._loc)
            f.attrs["kT"] = float(self._kbt)
            f.attrs["rng_bit_generator"] = type(self._rng.bit_generator).__name__
            f.attrs["rng_state_json"] = json.dumps(
                self._rng.bit_generator.state, default=json_default
            )

        return self

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        """Load system data from an HDF5 file.

        Args:
            filepath: Path to the file to load the data from

        Raises:
            ValueError: On missing/invalid checkpoint fields or unknown RNG class
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

        def attr_any(f: h5py.File, key: str) -> Any:
            return cast(Any, f.attrs[key])

        def attr_float(f: h5py.File, key: str) -> float:
            return float(attr_any(f, key))

        def attr_mesh(f: h5py.File) -> tuple[int, int]:
            key = "mesh"
            arr = np.asarray(attr_any(f, key))
            if arr.shape != (2,):
                raise ValueError(f"Expected {key} shape (2,), got {arr.shape}")
            return (int(arr[0]), int(arr[1]))

        def attr_f_loc(f: h5py.File) -> tuple[float, float]:
            key = "f_loc"
            arr = np.asarray(attr_any(f, key))
            if arr.shape != (2,):
                raise ValueError(f"Expected {key} shape (2,), got {arr.shape}")
            return (float(arr[0]), float(arr[1]))

        with h5py.File(filepath, "r") as f:
            mesh = attr_mesh(f)

            lam = attr_float(f, "lambda")
            mu = attr_float(f, "mu")
            f_ext = attr_float(f, "f_ext")
            f_loc = attr_f_loc(f)
            kbt = attr_float(f, "kT")

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
                f_ext=f_ext,
                f_loc=f_loc,
                kbt=kbt,
                state=state,
                rng=rng,
            )
