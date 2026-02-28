"""Spherical body subject to fluctuating forces and point-source surface forces."""

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
# def to_cartesian(arg: Point | Grid) -> Point | Grid:
#     """Transform spherical coordinates to the Cartesian."""
#     xi = arg[0] if isinstance(arg, tuple) else arg[:, 0]
#     theta = arg[1] if isinstance(arg, tuple) else arg[:, 1]
#     phi = arg[2] if isinstance(arg, tuple) else arg[:, 2]

#     x = xi * np.sin(theta) * np.cos(phi)
#     y = xi * np.sin(theta) * np.sin(phi)
#     z = xi * np.cos(theta)
#     return np.array([x, y, z]) if instance(arg, Real) else np.column_stack((x, y, z))


class System:
    """Elastodynamic system."""

    def __init__(
        self,
        mesh: tuple[int, int],
        lam: Real,
        mu: Real,
        f_ext: Real = 0.0,
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
            kbt: Effective temperature in units of energy defaults to one
            state: Current state of the system defaults to zeros
            rng: Optional state of the random number generator or seed
        """
        self._lam = lam
        self._mu = mu
        self._ext = f_ext
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

        sz = len(self._inter)
        lhs = np.einsum(
            "np,kqnp,mn,m->kqnp",
            self._inter._wts.reshape(mesh),  # np
            self._inter._bfs.reshape(2 * mesh),  # kqnp
            self._inter._radial_values,  # mn
            cast(NDArray, [poly(1.0) for poly in self._inter._radial_functions]),  # m
        ).reshape(sz, sz)
        self._inverse_lhs = np.linalg.inv(lhs)

        transform = self._inter.transform()
        partial = np.asarray(
            [self._inter.partial_x(), self._inter.partial_y(), self._inter.partial_z()]
        )
        self._grad_div = np.einsum("aqk,bkl,ln->aqbn", partial, partial, transform)
        self._laplacian = np.einsum("bqk,bkl,ln->qn", partial, partial, transform)

    @property
    def __len__(self):
        """Return number of grid points."""
        return len(self._inter)

    @property
    def mesh(self) -> tuple[int, int]:
        """Return mesh specification of the point grid."""
        return self._inter.mesh

    def points(self) -> Iterator[NDArray[np.floating]]:
        """Return generator of quadrature points."""
        return self._inter.points()

    def grid(self) -> Grid:
        """Return generator of quadrature points."""
        return self._inter.grid()

    # def _elastic_stress(self):
    # (self._lam + self._mu)
    # term1 = np.einsum("aqk,bkl,ln,nb->aq", partial, partial, transform, self._state)
    # term2 = np.einsum("bqk,bkl,ln,na->aq", partial, partial, transform, self._state)

    # return (self._lam + self._mu) * term1 + self._mu * term2

    # def _external_force(self):
    #     chi_north_pole = np.asarray(
    #         [chi([0.0, 0.0]) for chi in self._inter.angular_functions]
    #     )

    #     return (
    #         self._f_ext
    #         * (self._radial_surface[:, None] * chi_north_pole[None, :]).ravel()
    #     )

    # def _fluctuations(self, dt):
    #     normal = self.rng.normal(0.0, 1.0, size=(3, self._inter.nangles))

    #     return np.asarray(
    #         [
    #             np.sqrt(2 * self._kbt * self._gamma * dt)
    #             * (self._radial_surface[:, None] * nor[None, :]).ravel()
    #             for nor in normal
    #         ]
    #     )

    def __call__(self, dt: Real) -> Self:
        """Euler-Maruyama integration step.

        Args:
            dt: Integration time step

        Returns:
            Updated system after one integration step.
        """
        # elastic_stress = self._elastic_stress()
        # external_force = self._external_force()
        # fluct = self._fluctuations(dt)
        div_stress = (self._mu + self._lam) * np.einsum(
            "aqbn,nb->qa", self._grad_div, self._state
        ) + self._mu * np.einsum("qn,na->qa", self._laplacian, self._state)
        delta = np.einsum("nq,qa->na", self._inverse_lhs, -div_stress)
        self._state = np.array(self._state) + dt * delta
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

        def attr_mesh2(f: h5py.File, key: str = "mesh") -> tuple[int, int]:
            arr = np.asarray(attr_any(f, key))
            if arr.shape != (2,):
                raise ValueError(f"Expected {key} shape (2,), got {arr.shape}")
            return (int(arr[0]), int(arr[1]))

        with h5py.File(filepath, "r") as f:
            mesh = attr_mesh2(f)

            lam = attr_float(f, "lambda")
            mu = attr_float(f, "mu")
            f_ext = attr_float(f, "f_ext")
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
                mesh=mesh, lam=lam, mu=mu, f_ext=f_ext, kbt=kbt, state=state, rng=rng
            )
