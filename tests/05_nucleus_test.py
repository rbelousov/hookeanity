"""Analytic OU transitions and compatibility of the exact nucleus simulator."""

import inspect
import json
import xml.etree.ElementTree as ET
from typing import TypedDict

import h5py
import hookeanity
import numpy as np
import pytest
from hookeanity.nucleus import System, _covariance_factor, _transition_moments
from hookeanity.system import AndularGrid
from hookeanity.system import System as SpatialSystem
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad_vec
from scipy.linalg import expm
from scipy.spatial.transform import Rotation


class SpatialParameters(TypedDict):
    """Constructor parameters shared by the spatial and exact systems."""

    resolution: tuple[int, int]
    lam: float
    mu: float
    dt: float
    f_ext: float
    f_loc: tuple[float, float]
    kbt: float
    gamma: float
    angular_grid: AndularGrid
    lmax: int


class ToySystem(System):
    """Three surface DOFs with a known drift and a simple bulk constraint."""

    def _build_operators(self):
        self._mesh = (2, 1)
        self._grid = np.array([[0.5, 1.0, 0.0], [1.0, 1.0, 0.0]])
        self._trac_red = np.diag([-0.3, -1.0, -2.0])
        self._bulk_from_surf = 0.5 * np.eye(3)
        self._ela = np.column_stack((np.eye(3), -self._bulk_from_surf))
        self._trac = np.column_stack((np.zeros((3, 3)), self._trac_red))
        self._force = self._ext * np.array([1.0, -0.5, 0.25])

    def _build_rotation_projection(self):
        """A single surface point has no identifiable global rotation."""
        self._rotation_fit = np.zeros((3, self.nsurf))
        self._surface_rotation_modes = np.zeros((self.nsurf, 3))


def toy(**kwargs):
    """Factory of a toy system."""
    return ToySystem((2, 1), **{"lam": 1.0, "mu": 1.0, "dt": 0.2, **kwargs})


@pytest.mark.parametrize("dt", [0.0, 1e-10, 0.7, 10.0])
def test_diagonal_transition(dt):
    """First-smoke test."""
    rates = np.array([-2.0, -0.3, 0.0])
    force = np.array([0.7, -1.2, 0.5])
    diffusion = 0.8
    transition, mean, covariance = _transition_moments(
        np.diag(rates), force, diffusion, dt
    )
    integral = np.full(3, dt)
    integral[:2] = np.expm1(rates[:2] * dt) / rates[:2]
    variance = np.full(3, diffusion * dt)
    variance[:2] = diffusion * np.expm1(2 * rates[:2] * dt) / (2 * rates[:2])
    assert_allclose(transition, np.diag(np.exp(rates * dt)), atol=1e-14)
    assert_allclose(mean, integral * force, atol=1e-14)
    assert_allclose(covariance, np.diag(variance), atol=1e-14)


@pytest.mark.parametrize(
    "drift",
    [
        np.array([[-0.7]]),
        np.zeros((2, 2)),
        np.array([[-1.0, 2.0], [0.0, -1.0]]),  # defective
        np.array([[-0.4, 1.0], [-2.0, -0.7]]),  # complex eigenvalues
        np.array([[0.0, 1.0], [0.0, 0.0]]),  # singular and defective
    ],
)
def test_transition_against_quadrature(drift):
    """Test Van Loan exponential and quadrature of the matrix exponential agree."""
    n = len(drift)
    force = np.linspace(-0.2, 0.6, n)
    dt, diffusion = 1.3, 0.9
    transition, mean, covariance = _transition_moments(drift, force, diffusion, dt)
    expected_mean = quad_vec(lambda t: expm(drift * t) @ force, 0, dt)[0]
    expected_cov = quad_vec(
        lambda t: diffusion * expm(drift * t) @ expm(drift.T * t), 0, dt
    )[0]
    assert_allclose(transition, expm(drift * dt), atol=1e-13)
    assert_allclose(mean, expected_mean, atol=1e-13)
    assert_allclose(covariance, expected_cov, atol=1e-13)


def test_stiff_transition():
    """A direct Van Loan exponential over dt=1 would overflow its +10000 block."""
    transition, mean, covariance = _transition_moments(
        np.diag([-1e4, 0.0]), np.array([0.2, -0.3]), 2.0, 1.0
    )
    assert_allclose(transition, np.diag([0.0, 1.0]), atol=1e-14)
    assert_allclose(mean, [2e-5, -0.3], atol=1e-14)
    assert_allclose(covariance, np.diag([1e-4, 2.0]), atol=1e-14)


def test_transition_composition():
    """Test transition components."""
    drift = np.array([[-0.1, 3.0], [0.0, -2.0]])
    force = np.array([0.5, -0.2])
    f, m, q = _transition_moments(drift, force, 0.7, 2.4)
    f1, m1, q1 = _transition_moments(drift, force, 0.7, 0.9)
    f2, m2, q2 = _transition_moments(drift, force, 0.7, 1.5)
    assert_allclose(f, f2 @ f1, atol=1e-13)
    assert_allclose(m, f2 @ m1 + m2, atol=1e-13)
    assert_allclose(q, f2 @ q1 @ f2.T + q2, atol=1e-13)


@pytest.mark.parametrize("scale", [1.0, 1e-20])
def test_covariance_factor_roundoff(scale):
    """Test noise amplitude."""
    q = scale * np.array([[1.0, 0.3], [0.3, 0.5]])
    factor = _covariance_factor(q)
    assert_allclose(factor @ factor.T, q, rtol=1e-14, atol=0)
    factor = _covariance_factor(scale * np.diag([1.0, -1e-15]))
    assert_allclose(factor @ factor.T, scale * np.diag([1.0, 0.0]), atol=0)
    with pytest.raises(ValueError, match="positive semidefinite"):
        _covariance_factor(scale * np.diag([1.0, -1e-3]))


def test_sampled_transition():
    """Test random transitions."""
    sys = toy(f_ext=0.6, kbt=0.4, gamma=1.7, rng=31)
    # Correlated noise must be used for a nonsymmetric coupled drift.
    sys._trac_red[0, 1] = 1.2
    sys._build_transition()
    initial = np.array([[0.4, -0.3, 0.2], [0.1, 0.5, -0.2]])
    drift = sys._trac_red / sys._gamma
    expected_mean = (
        expm(drift * sys._dt) @ initial[-1]
        + quad_vec(lambda t: expm(drift * t) @ (sys._force / sys._gamma), 0, sys._dt)[0]
    )
    expected_cov = quad_vec(
        lambda t: 2 * sys._kbt / sys._gamma * expm(drift * t) @ expm(drift.T * t),
        0,
        sys._dt,
    )[0]
    count = 30000
    samples = np.empty((count, 3))
    for i in range(count):
        sys._state = initial.copy()
        assert sys() is sys
        samples[i] = sys.surface()[0]
    mean_error = np.abs(samples.mean(axis=0) - expected_mean)
    cov_error = np.abs(np.cov(samples, rowvar=False) - expected_cov)
    assert np.all(mean_error < 6 * np.sqrt(np.diag(expected_cov) / count))
    cov_standard_error = np.sqrt(
        (np.outer(np.diag(expected_cov), np.diag(expected_cov)) + expected_cov**2)
        / (count - 1)
    )
    assert np.all(cov_error < 6 * cov_standard_error)


@pytest.mark.parametrize("dt,kbt", [(0.0, 1.0), (0.2, 0.0)])
def test_deterministic_and_zero_steps(dt, kbt):
    """Test deterministic system without noise."""
    initial = np.arange(6.0).reshape(2, 3)
    sys = toy(dt=dt, kbt=kbt, gamma=2.3, f_ext=0.7, rng=12, state=initial)
    assert_array_equal(sys.state(), initial)
    rng_state = json.dumps(sys._rng.bit_generator.state)
    assert sys() is sys
    assert json.dumps(sys._rng.bit_generator.state) == rng_state
    if dt == 0:
        assert_array_equal(sys.state(), initial)
        assert_array_equal(sys.rotation(), np.eye(3))
    else:
        rates = np.diag(sys._trac_red) / sys._gamma
        expected = np.exp(rates * dt) * initial[-1]
        expected += np.expm1(rates * dt) / rates * sys._force / sys._gamma
        assert_allclose(sys.surface()[0], expected)
        assert_allclose(sys._ela @ sys.state().ravel(), 0, atol=1e-14)


def test_interface():
    """Test compatibility with `hookeanity.system.System`."""
    expected = dict(inspect.signature(SpatialSystem).parameters)
    expected.pop("method")
    assert dict(inspect.signature(System).parameters) == expected
    assert hookeanity.System is SpatialSystem
    assert not hasattr(toy(), "_method")
    with pytest.raises(TypeError, match="method"):
        toy(method="exact")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": -1},
        {"gamma": 0},
        {"gamma": -1},
        {"kbt": -1},
        {"dt": np.inf},
        {"lam": np.nan},
        {"mu": np.inf},
        {"gamma": np.nan},
        {"kbt": np.inf},
        {"f_ext": np.nan},
        {"f_loc": (0.0, np.inf)},
        {"state": np.zeros((1, 3))},
        {"state": np.full((2, 3), np.nan)},
        {"angular_grid": "unknown"},
        {"rotation": np.diag([-1.0, 1.0, 1.0])},
        {"rotation": np.zeros((3, 3))},
    ],
)
def test_invalid_parameters(kwargs):
    """Test exceptions."""
    with pytest.raises(ValueError):
        toy(**kwargs)


@pytest.fixture(
    scope="module",
    params=[
        ("fibonacci", (5, 12), 5),
        ("gauss-legendre-fourier", (4, 5), 15),
    ],
)
def spatial_data(request, tmp_path_factory):
    """Spatial system with various grids."""
    grid, resolution, lmax = request.param
    params = SpatialParameters(
        resolution=resolution,
        lam=1.0,
        mu=1.0,
        dt=0.1,
        f_ext=0.2,
        f_loc=(0.7, 0.5),
        kbt=0.3,
        gamma=1.7,
        angular_grid=grid,
        lmax=lmax,
    )
    path = tmp_path_factory.mktemp(grid) / "spatial.h5"
    spatial = SpatialSystem(**params)
    spatial.save_cache(path)
    return params, path, spatial


def test_spatial_reduction_and_rigid_modes(spatial_data):
    """Test reduction of bulk degrees of freedom."""
    params, _, _ = spatial_data
    sys = System(**params, rng=42)
    raw = sys._trac[:, -sys.nsurf :] + sys._trac[:, : -sys.nsurf] @ sys._bulk_from_surf
    modes = sys._surface_rigid_modes()
    q, _ = np.linalg.qr(modes)
    p = np.eye(sys.nsurf) - q @ q.T
    assert_allclose(sys._trac_red, p @ raw @ p, atol=1e-10)
    assert_allclose(
        q.T @ sys._mean, sys._dt / sys._gamma * q.T @ sys._force, atol=1e-12
    )
    assert_allclose(
        q.T @ sys._covariance @ q,
        2 * sys._kbt * sys._dt / sys._gamma * np.eye(q.shape[1]),
        atol=1e-12,
    )
    assert sys()()() is sys
    assert_allclose(sys._ela @ sys.state().ravel(), 0, atol=1e-8)


def test_rotation_projection(spatial_data):
    """The reconstructed internal OU noise has no full-body rotation."""
    params, cache, _ = spatial_data
    sys = System(**params, cache=cache, rng=42)
    omega = np.array([0.01, -0.02, 0.03])
    internal, extracted = sys._split_noise(sys._surface_rotation_modes @ omega)
    assert_allclose(internal, 0, atol=1e-12)
    assert_allclose(extracted, omega, atol=1e-12)

    noise = sys._noise_factor @ np.random.default_rng(7).normal(size=sys.nsurf)
    internal, _ = sys._split_noise(noise)
    centered = sys._reference_xyz - sys._center
    modes = np.stack([np.cross(np.eye(3)[a], centered) for a in range(3)], axis=-1)
    residual = sys._reconstruct_state(internal)
    overlap = np.einsum("p,pai,pa->i", sys._volume_weights, modes, residual)
    assert_allclose(overlap, 0, atol=1e-10)


def test_pure_rotational_ou_step(spatial_data, monkeypatch):
    """A pure rotational OU draw changes orientation without internal strain."""
    params, _, _ = spatial_data
    sys = System(**{**params, "f_ext": 0.0}, rng=42)
    omega = np.array([0.01, -0.02, 0.03])
    sys._noise_factor[:] = 0.0
    sys._noise_factor[:, 0] = sys._surface_rotation_modes @ omega

    class FixedNoise:
        def normal(self, size):
            assert size == sys.nsurf
            value = np.zeros(size)
            value[0] = 1.0
            return value

    monkeypatch.setattr(sys, "_rng", FixedNoise())
    sys()
    assert_allclose(sys.state(), 0, atol=1e-11)
    assert_allclose(sys.rotation(), Rotation.from_rotvec(omega).as_matrix(), atol=1e-12)


def test_rotation_composition_and_force_mean(spatial_data):
    """Use start-of-step orientation for the exact frozen-frame force response."""
    params, cache, _ = spatial_data
    start = Rotation.from_rotvec([0.0, np.pi / 2, 0.0]).as_matrix()
    sys = System(**{**params, "kbt": 0.0}, rotation=start, cache=cache)
    force_body = sys._body_force() / sys._gamma
    expected = quad_vec(
        lambda t: expm(sys._trac_red * (t / sys._gamma)) @ force_body,
        0,
        sys._dt,
    )[0]
    sys()
    assert_allclose(sys.surface().ravel(), expected, atol=1e-11)
    assert_allclose(sys.rotation(), start, atol=1e-12)

    stochastic = System(**{**params, "f_ext": 0.0}, rotation=start, rng=123)
    random = np.random.default_rng(123)
    expected_rotation = start.copy()
    for _ in range(3):
        noise = stochastic._noise_factor @ random.normal(size=stochastic.nsurf)
        omega = stochastic._rotation_fit @ noise
        expected_rotation = expected_rotation @ Rotation.from_rotvec(omega).as_matrix()
        stochastic()
    assert_allclose(stochastic.rotation(), expected_rotation, atol=1e-12)
    assert_allclose(
        stochastic.rotation().T @ stochastic.rotation(), np.eye(3), atol=1e-12
    )
    assert np.linalg.det(stochastic.rotation()) == pytest.approx(1.0)


@pytest.mark.parametrize("bitgen", [np.random.PCG64, np.random.MT19937])
def test_checkpoint_continuation(spatial_data, tmp_path, bitgen):
    """Test checkpointing and continuation of the exact nucleus simulator."""
    params, cache, _ = spatial_data
    rotation = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_matrix()
    sys = System(
        **params, cache=cache, rng=np.random.Generator(bitgen(42)), rotation=rotation
    )
    sys()()
    checkpoint = tmp_path / "nucleus.h5"
    assert sys.save(checkpoint) is sys
    with h5py.File(checkpoint, "r") as f:
        assert "method" not in f.attrs
        assert "init" not in f.attrs
        assert "rotation" in f
    loaded = System.load(checkpoint, cache=cache)
    assert type(loaded) is System
    assert not hasattr(loaded, "_method")
    assert_array_equal(sys.state(), loaded.state())
    assert_allclose(sys.rotation(), loaded.rotation())
    for _ in range(3):
        assert_array_equal(sys().state(), loaded().state())
        assert_array_equal(sys.rotation(), loaded.rotation())


def test_spatial_checkpoint_and_cache(spatial_data, tmp_path):
    """Load a current spatial-system checkpoint and cache."""
    params, cache, _ = spatial_data
    rotation = Rotation.from_rotvec([0.15, 0.05, -0.2]).as_matrix()
    spatial = SpatialSystem(**params, rotation=rotation)
    checkpoint = tmp_path / "spatial.h5"
    spatial.save(checkpoint)
    loaded = System.load(checkpoint, cache=cache)
    direct = System(**params, cache=cache)
    assert_array_equal(loaded._force, spatial._force)
    assert_array_equal(loaded._trac_red, spatial._trac_red)
    assert_array_equal(loaded._transition, direct._transition)
    assert not hasattr(loaded, "_method")
    assert_array_equal(loaded.state(), spatial.state())
    assert_allclose(loaded.rotation(), rotation)
    assert_array_equal(loaded._rng.random(5), spatial._rng.random(5))


def test_cache_rebuild_and_validation(spatial_data, tmp_path):
    """Test cache data."""
    params, cache, _ = spatial_data
    sys = System(**params, cache=cache, rng=1)
    # gamma and kbt are not spatial-cache parameters and must be applied afresh.
    other_params = {**params, "gamma": 3.0, "kbt": 0.1}
    other = System(**other_params, cache=cache, rng=1)
    assert not np.allclose(sys._transition, other._transition)
    assert not np.allclose(sys._covariance, other._covariance)
    own_cache = tmp_path / "nucleus-cache.h5"
    assert sys.save_cache(own_cache) is sys
    with h5py.File(own_cache, "r+") as f:
        assert f.attrs["rigid_projection"] == "all"
        assert "init" not in f.attrs
        f.attrs["rigid_projection"] = "no_rotations"
    with pytest.raises(ValueError, match="incompatible rigid projection"):
        System(**params, cache=own_cache)
    with h5py.File(own_cache, "r+") as f:
        f.attrs["rigid_projection"] = "all"
    other._transition[:] = np.nan
    other._mean[:] = np.nan
    other._covariance[:] = np.nan
    other._noise_factor[:] = np.nan
    assert other.load_cache(own_cache) is other
    rebuilt = System(**other_params, cache=own_cache, rng=1)
    assert_array_equal(other().state(), rebuilt().state())
    with pytest.raises(AssertionError, match="time steps"):
        System(**{**params, "dt": 0.2}, cache=cache)
    with h5py.File(own_cache, "r+") as f:
        traction = f["trac_red"]
        assert isinstance(traction, h5py.Dataset)
        traction[...] = np.eye(sys.nsurf)
    with pytest.raises(AssertionError, match="Unstable mode"):
        sys.load_cache(own_cache)


def test_exports(spatial_data, tmp_path):
    """Test VTS and VTP exports."""
    params, cache, _ = spatial_data
    start = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_matrix()
    sys = System(**params, cache=cache, rng=42, rotation=start)()
    projection = tmp_path / "projection.h5"
    assert sys.save_projection_cache(projection, resolution=7) is sys
    assert sys.load_projection_cache(projection, resolution=7).shape == (
        49,
        sys.mesh[1],
    )
    direct, cached, native = (
        tmp_path / name for name in ["direct.vts", "cached.vts", "native.vtp"]
    )
    assert sys.export_vts(direct, resolution=7) is sys
    assert sys.export_vts(cached, resolution=7, projection_cache=projection) is sys
    direct_arrays = list(ET.parse(direct).getroot().iter("DataArray"))
    cached_arrays = list(ET.parse(cached).getroot().iter("DataArray"))
    assert len(direct_arrays) == len(cached_arrays)
    for expected, actual in zip(direct_arrays, cached_arrays):
        assert expected.attrib == actual.attrib
        assert actual.text is not None
        assert expected.text is not None
        assert_allclose(
            np.fromstring(actual.text, sep=" "),
            np.fromstring(expected.text, sep=" "),
            atol=1e-12,
        )
    dense_root = ET.parse(direct).getroot()
    dense_points_array = dense_root.find(".//Points/DataArray")
    dense_vector_array = dense_root.find(".//DataArray[@Name='Displacement_Vector']")
    assert dense_points_array is not None and dense_points_array.text is not None
    assert dense_vector_array is not None and dense_vector_array.text is not None
    dense_points = np.fromstring(dense_points_array.text, sep=" ").reshape(-1, 3)
    dense_vectors = np.fromstring(dense_vector_array.text, sep=" ").reshape(-1, 3)
    theta, phi = np.meshgrid(np.linspace(0, np.pi, 7), np.linspace(0, 2 * np.pi, 7))
    radius = np.mean(sys.surface_points()[:, 0])
    dense_reference = sys._spherical_to_cartesian(
        np.column_stack((np.full(49, radius), theta.ravel(), phi.ravel()))
    )
    dense_internal = sys._build_vts_projection(7) @ sys.surface()
    expected_dense = (
        sys._center
        + (dense_reference + dense_internal - sys._center) @ sys.rotation().T
    )
    assert_allclose(dense_points, expected_dense, atol=1e-12)
    assert_allclose(dense_vectors, dense_points - dense_reference, atol=1e-12)
    assert sys.export_vtp(native) is sys
    root = ET.parse(native).getroot()
    assert root.attrib["type"] == "PolyData"
    vector = root.find(".//DataArray[@Name='Displacement_Vector']")
    points_array = root.find(".//Points/DataArray")
    assert vector is not None
    assert points_array is not None
    assert vector.text is not None
    assert points_array.text is not None
    points = np.fromstring(points_array.text, sep=" ").reshape(-1, 3)
    reference = sys._spherical_to_cartesian(sys.surface_points())
    expected_points = (
        sys._center + (reference + sys.surface() - sys._center) @ sys.rotation().T
    )
    assert_allclose(points, expected_points, atol=1e-12)
    assert_allclose(
        np.fromstring(vector.text, sep=" ").reshape(-1, 3), points - reference
    )


@pytest.mark.parametrize("cls", [System, SpatialSystem])
def test_optional_lmax_roundtrip(cls, tmp_path):
    """Ensure handling of the optional `lmax` cache entries."""
    sys = cls((5, 12), 1.0, 1.0, 0.01, rng=42)
    checkpoint, cache, projection = (
        tmp_path / name for name in ["state.h5", "cache.h5", "projection.h5"]
    )
    sys.save(checkpoint).save_cache(cache).save_projection_cache(
        projection, resolution=5
    )
    loaded = cls.load(checkpoint, cache=cache)
    assert loaded.lmax is None
    assert_array_equal(loaded.state(), sys.state())
    assert_array_equal(loaded().state(), sys().state())
    assert_allclose(
        loaded.load_projection_cache(projection, resolution=5),
        sys._build_vts_projection(5),
    )
