"""Tests for `hookeanity.system` module.

Attributes:
    GRID_CONFIGS: Grid specifications for solid angles.
"""

import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest
from hookeanity.base import DIM
from hookeanity.interpolation import Interpolation
from hookeanity.system import System
from scipy.sparse.linalg import expm_multiply
from scipy.spatial.transform import Rotation

GRID_CONFIGS = [
    ("fibonacci", (5, 12), 60, 5),
    ("gauss-legendre-fourier", (4, 5), 200, 15),
]


@pytest.fixture(
    scope="module",
    params=[
        ("fibonacci", (5, 12), 60, 5),
        ("gauss-legendre-fourier", (4, 5), 200, 15),
    ],
)
def system(request):
    """Return `System` object for different grids, with a random initial state."""
    angular_grid, resolution, npoints, lmax = request.param
    state = np.random.default_rng(0).uniform(low=-1e-3, high=1e-3, size=(npoints, DIM))
    sys = System(
        resolution,
        lam=1.0,
        mu=1.0,
        dt=1e-2,
        lmax=lmax,
        angular_grid=angular_grid,
        kbt=0.0,
        state=state,
    )
    return sys


@pytest.fixture(
    scope="module",
    params=[
        ("fibonacci", (5, 12), 5),
        ("gauss-legendre-fourier", (4, 5), 15),
    ],
)
def system_zero_state(request):
    """Return System object for different grids, with the default inital state."""
    angular_grid, resolution, lmax = request.param
    sys = System(
        resolution,
        lam=1.0,
        mu=1.0,
        dt=1e-2,
        lmax=lmax,
        angular_grid=angular_grid,
        kbt=0.0,
    )
    return sys


@pytest.mark.parametrize("angular_grid, resolution, npoints, lmax", GRID_CONFIGS)
def test_system_init(angular_grid, resolution, npoints, lmax):
    """Test `hookeanity.System` initialization."""
    sys = System(resolution, 1.0, 1.0, 1e-2, lmax=lmax, angular_grid=angular_grid)
    assert sys._state.shape == (npoints, DIM)
    assert np.all(sys._state == 0)


@pytest.mark.parametrize("angular_grid, resolution, npoints, lmax", GRID_CONFIGS)
def test_save_and_load(tmp_path, angular_grid, resolution, npoints, lmax):
    """Test saving and loading system data in the HDF5 format."""
    filepath = tmp_path / "system.h5"

    lam, mu, dt = 2.25, 2.0, 1e-2
    gamma, f_ext = 2.5, 0.5
    f_loc = (0.7, 1.2)
    kbt = 1.5
    method = "trotter"
    rotation = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_matrix()

    state = np.random.default_rng(0).uniform(low=-1e-3, high=1e-3, size=(npoints, DIM))

    system = System(
        resolution,
        lam,
        mu,
        dt,
        f_ext,
        f_loc=f_loc,
        kbt=kbt,
        gamma=gamma,
        state=state,
        method=method,
        rng=123,
        lmax=lmax,
        angular_grid=angular_grid,
        rotation=rotation,
    )

    system.save(filepath)
    sample1 = system._rng.random(5)

    assert system.resolution == resolution
    assert system.mesh == (resolution[0], int(npoints / resolution[0]))
    assert system.lmax == lmax
    assert system.surface().shape == (system.mesh[1], DIM)
    assert system.surface_points().shape == (system.mesh[1], DIM)

    loaded = System.load(filepath)
    sample2 = loaded._rng.random(5)
    assert loaded.resolution == resolution
    assert loaded.mesh == (resolution[0], int(npoints / resolution[0]))
    assert loaded._lam == lam
    assert loaded._mu == mu
    assert loaded._dt == dt
    assert loaded._gamma == gamma
    assert loaded._ext == f_ext
    assert loaded._loc == f_loc
    assert loaded._kbt == kbt
    assert loaded.lmax == lmax
    assert loaded._method == method
    assert loaded._state == pytest.approx(state)
    assert loaded.rotation() == pytest.approx(rotation)
    assert np.all(sample1 == sample2)
    system()
    loaded()
    assert loaded.state() == pytest.approx(system.state())
    assert loaded.rotation() == pytest.approx(system.rotation())


def test_stress_divergence(system):
    """Test formulas for the divergence of stress."""
    sys = system

    inter = Interpolation(sys.resolution, angular_grid=sys._angular_grid, lmax=sys.lmax)
    transform = inter.transform()
    inverse = inter.inverse()
    partial = np.asarray([inter.partial_x(), inter.partial_y(), inter.partial_z()])

    grad_div = np.einsum(
        "lL,aLM,bMN,Nn->lanb", inverse, partial, partial, transform, optimize=True
    )
    laplacian = np.einsum(
        "lL,bLM,bMN,Nn->ln", inverse, partial, partial, transform, optimize=True
    )
    expected = (
        (sys._mu + sys._lam)
        * np.einsum("lanb,nb->la", grad_div, sys._state, optimize=True)
        + sys._mu * np.einsum("ln,na->la", laplacian, sys._state, optimize=True)
    ).ravel()

    ndof = sys.ndof
    gd = grad_div.reshape(ndof, ndof)
    lap = np.einsum(
        "ab,lL,cLM,cMN,Nn->lanb",
        np.eye(DIM),
        inverse,
        partial,
        partial,
        transform,
        optimize=True,
    ).reshape(ndof, ndof)
    mtx = (sys._mu + sys._lam) * gd + sys._mu * lap
    value = mtx @ sys._state.ravel()
    assert value == pytest.approx(expected)

    value = sys._ela @ sys._state.ravel()
    assert value == pytest.approx(expected[: -DIM * sys.mesh[1]])


def test_traction(system):
    """Test formulas for the traction force."""
    sys = system
    inter = Interpolation(sys.resolution, angular_grid=sys._angular_grid, lmax=sys.lmax)
    mesh = sys.mesh
    eye = np.eye(DIM)
    transform = inter.transform()
    inverse = inter.inverse()
    partial_xyz = np.array([inter.partial_x(), inter.partial_y(), inter.partial_z()])
    partial = np.einsum(
        "mM,aMN,Nn->man", inverse, partial_xyz, transform, optimize=True
    )

    normals = sys.normals()
    nabla = np.einsum("pan,nb->pab", partial[-mesh[1] :, :, :], sys._state)
    div = np.trace(nabla, axis1=1, axis2=2)
    sym = nabla + np.swapaxes(nabla, 1, 2)
    stress = sys._lam * div[:, None, None] * eye[None, :, :] + sys._mu * sym
    expected = np.einsum("pa,pab->pb", normals, stress).ravel()

    term0 = np.einsum("pa,pan->pn", normals, partial[-mesh[1] :, :, :])
    term1 = sys._mu * np.einsum("pn,bc->pbnc", term0, eye)
    term2 = sys._lam * np.einsum("pb,pcn->pbnc", normals, partial[-mesh[1] :, :, :])
    term3 = sys._mu * np.einsum("pc,pbn->pbnc", normals, partial[-mesh[1] :, :, :])
    trac = (term1 + term2 + term3).reshape(DIM * mesh[1], DIM * np.prod(mesh))
    value = trac @ sys._state.ravel()
    assert value == pytest.approx(expected)

    value = sys._trac @ sys._state.ravel()
    assert value == pytest.approx(expected)


def test_system_smoke(system_zero_state):
    """Check first smoke of the system functionality."""
    sys = system_zero_state
    assert sys._kbt == 0.0
    assert sys()._state.shape == (system_zero_state.npoints, DIM)
    assert np.all(sys._state == 0)

    sys._state += 1
    assert np.all(sys._state == 1)
    assert sys()._state.shape == (system_zero_state.npoints, DIM)
    assert sys._state == pytest.approx(1, abs=1e-3)

    sys = System(
        system_zero_state.resolution,
        1.0,
        1.0,
        1e-2,
        verbose=False,
        lmax=system_zero_state.lmax,
    )()
    assert sys._kbt == 1
    assert sys._state != pytest.approx(1, abs=1e-3)

    sys_n = System(
        system_zero_state.resolution,
        1.0,
        1.0,
        1e-2,
        f_ext=10,
        f_loc=(0, 0),
        kbt=0.0,
        lmax=system_zero_state.lmax,
    )
    sys_s = System(
        system_zero_state.resolution,
        1.0,
        1.0,
        1e-2,
        f_ext=10,
        f_loc=(np.pi, 0),
        kbt=0.0,
        lmax=system_zero_state.lmax,
    )
    assert sys_n._force != pytest.approx(sys_s._force)
    sys_n()()()
    sys_s()()()
    assert sys_n._state != pytest.approx(sys_s._state)

    sys_trotter = System(
        system_zero_state.resolution,
        1.0,
        1.0,
        1e-2,
        f_ext=10,
        f_loc=(np.pi, 0),
        kbt=0.0,
        method="trotter",
        lmax=system_zero_state.lmax,
    )
    sys_trotter()
    assert np.isfinite(sys_trotter._state).all()

    sys_strang = System(
        system_zero_state.resolution,
        1.0,
        1.0,
        1e-2,
        f_ext=10,
        f_loc=(0, 0),
        kbt=0.0,
        method="strang",
        lmax=system_zero_state.lmax,
    )
    sys_strang()
    assert np.isfinite(sys_strang._state).all()


@pytest.mark.parametrize("angular_grid, resolution, npoints, lmax", GRID_CONFIGS)
def test_system_cache(tmp_path, angular_grid, resolution, npoints, lmax):
    """Test saving and loading cache data for `System` objects."""
    cache = tmp_path / "cache.5x6.l2m3.h5"
    sys1 = System(
        resolution,
        2.0,
        2.0,
        1e-2,
        lmax=lmax,
        angular_grid=angular_grid,
    )
    sys1.save_cache(cache)
    sys2 = System(
        resolution,
        2.0,
        2.0,
        1e-2,
        lmax=lmax,
        cache=cache,
        angular_grid=angular_grid,
    )
    assert sys1.npoints == npoints
    assert sys2.npoints == npoints

    assert sys2.lmax == sys1.lmax
    assert np.all(sys1._grid == sys2._grid)
    assert np.all(sys1._trac == sys2._trac)
    assert np.all(sys1._ela == sys2._ela)
    assert np.all(sys1._bulk_from_surf == sys2._bulk_from_surf)
    assert np.all(sys1._trac_red == sys2._trac_red)
    assert np.all(sys1._force == sys2._force)
    assert sys1._volume_weights == pytest.approx(sys2._volume_weights)

    with h5py.File(cache, "r+") as f:
        del f.attrs["rigid_projection"]
    with pytest.raises(ValueError, match="rebuild the cache"):
        System(
            resolution,
            2.0,
            2.0,
            1e-2,
            lmax=lmax,
            cache=cache,
            angular_grid=angular_grid,
        )
    sys1.save_cache(cache)

    with pytest.raises(AssertionError, match="spherical-harmonic cutoffs"):
        System(resolution, 2.0, 2.0, 1e-2, cache=cache, angular_grid=angular_grid)

    with pytest.raises(AssertionError, match="time steps"):
        System(
            resolution,
            2.0,
            2.0,
            2e-2,
            lmax=lmax,
            cache=cache,
            angular_grid=angular_grid,
        )


@pytest.mark.parametrize("angular_grid, resolution, npoints, lmax", GRID_CONFIGS)
def test_projection_cache(tmp_path, angular_grid, resolution, npoints, lmax):
    """Test saving and loading VTS projection cache data."""
    sys = System(
        resolution,
        1.0,
        1.0,
        1e-2,
        kbt=0.0,
        lmax=lmax,
        angular_grid=angular_grid,
    )

    dense_resolution = 7
    cache = tmp_path / "projection.h5"
    assert sys.save_projection_cache(cache, resolution=dense_resolution) is sys

    expected = sys._build_vts_projection(dense_resolution)
    projection = sys.load_projection_cache(cache, resolution=dense_resolution)

    assert projection.shape == (dense_resolution * dense_resolution, sys.mesh[1])
    assert projection == pytest.approx(expected)

    with pytest.raises(AssertionError, match="VTS export resolutions"):
        sys.load_projection_cache(cache, resolution=dense_resolution + 1)

    with h5py.File(cache, "r+") as f:
        f.attrs["lmax"] = lmax + 1
    with pytest.raises(AssertionError, match="spherical-harmonic cutoffs"):
        sys.load_projection_cache(cache, resolution=dense_resolution)


@pytest.mark.parametrize("angular_grid, resolution, npoints, lmax", GRID_CONFIGS)
def test_projection_cache_grid_mismatch(
    tmp_path, angular_grid, resolution, npoints, lmax
):
    """Test projection cache rejects mismatched system-grid metadata."""
    sys = System(
        resolution,
        1.0,
        1.0,
        1e-2,
        kbt=0.0,
        lmax=lmax,
        angular_grid=angular_grid,
    )

    cache = tmp_path / "projection.h5"
    sys.save_projection_cache(cache, resolution=7)
    with h5py.File(cache, "r+") as f:
        f.attrs["resolution"] = (resolution[0], resolution[1] + 1)

    with pytest.raises(AssertionError, match="resolutions"):
        sys.load_projection_cache(cache, resolution=7)


def _build_operators_old(
    resolution: tuple[int, int],
    mesh: tuple[int, int],
    lam: float,
    mu: float,
    f_ext: float,
    f_loc: tuple[float, float],
    angular_grid: str,
    lmax: int,
):
    """Old unoptimized implementation of `System._build_operators`.

    Returns:
        Tuple `(grid, trac, ela, force)`.
    """
    ndof = DIM * int(np.prod(mesh))
    eye = np.eye(DIM)

    inter = Interpolation(resolution, lmax=lmax, angular_grid=angular_grid)
    grid = inter.grid()

    # Derivative operators
    transform = inter.transform()
    inverse = inter.inverse()
    partial_xyz = np.array([inter.partial_x(), inter.partial_y(), inter.partial_z()])
    partial = np.einsum(
        "mM,aMN,Nn->man", inverse, partial_xyz, transform, optimize=True
    )

    # Surface normals (same formula as System.normals())
    angles = grid[-mesh[1] :, 1:]
    st = np.sin(angles[:, 0])
    st2 = st**2
    ct = np.cos(angles[:, 0])
    sp = np.sin(angles[:, 1])
    cp = np.cos(angles[:, 1])
    normals = -np.stack([st2 * cp, st2 * sp, st * ct], axis=1)

    # Traction operator
    term0 = np.einsum("pa,pan->pn", normals, partial[-mesh[1] :, :, :])
    term1 = mu * np.einsum("pn,bc->pbnc", term0, eye)
    term2 = lam * np.einsum("pb,pcn->pbnc", normals, partial[-mesh[1] :, :, :])
    term3 = mu * np.einsum("pc,pbn->pbnc", normals, partial[-mesh[1] :, :, :])
    trac = (term1 + term2 + term3).reshape(DIM * mesh[1], ndof)

    # Divergence of the elastic stress
    nbulk = ndof - DIM * mesh[1]
    gd = np.einsum(
        "lL,aLM,bMN,Nn->lanb",
        inverse[: -mesh[1]],
        partial_xyz,
        partial_xyz,
        transform,
        optimize=True,
    ).reshape(nbulk, ndof)

    vl = np.einsum(
        "ab,lL,cLM,cMN,Nn->lanb",
        eye,
        inverse[: -mesh[1]],
        partial_xyz,
        partial_xyz,
        transform,
        optimize=True,
    ).reshape(nbulk, ndof)

    ela = (mu + lam) * gd + mu * vl

    # Additive forces
    force = np.zeros((mesh[1], DIM))
    force[:, 2] = f_ext * np.array(
        [chi(np.array(f_loc)) for chi in inter._angular_functions]
    )
    force = force.ravel()

    return trac, ela, force


@pytest.mark.parametrize("angular_grid, resolution, npoints, lmax", GRID_CONFIGS)
def test_build_operators(angular_grid, resolution, npoints, lmax):
    """Test that the optimized `System._build_operators()` matches the old implementation."""
    lam = 2.0
    mu = 2.0
    dt = 1e-2
    f_ext = 0.5
    f_loc = (0.7, 1.2)

    sys = System(
        resolution,
        lam,
        mu,
        dt,
        f_ext=f_ext,
        f_loc=f_loc,
        kbt=0.0,
        angular_grid=angular_grid,
        lmax=lmax,
    )
    trac_old, ela_old, force_old = _build_operators_old(
        resolution=resolution,
        mesh=sys.mesh,
        lam=lam,
        mu=mu,
        f_ext=f_ext,
        f_loc=f_loc,
        angular_grid=angular_grid,
        lmax=lmax,
    )

    assert sys._trac.shape == trac_old.shape
    assert sys._ela.shape == ela_old.shape
    assert sys._force.shape == force_old.shape

    assert sys._trac == pytest.approx(trac_old)
    assert sys._ela == pytest.approx(ela_old)
    assert sys._force == pytest.approx(force_old)


@pytest.mark.parametrize("angular_grid, resolution, npoints, lmax", GRID_CONFIGS)
def test_bulk_reconstruction(angular_grid, resolution, npoints, lmax):
    """Test resoncstruction of the bulk DoFs' displacements from the surface DoFs."""
    sys = System(
        resolution,
        2.0,
        2.0,
        1e-2,
        f_ext=0.5,
        f_loc=(0.7, 1.2),
        kbt=0.0,
        angular_grid=angular_grid,
        lmax=lmax,
    )

    surf = np.random.default_rng(0).uniform(low=-1e-3, high=1e-3, size=sys.nsurf)
    bulk = -np.linalg.solve(
        sys._ela[:, : -sys.nsurf],
        sys._ela[:, -sys.nsurf :] @ surf,
    )

    assert sys._bulk_from_surf @ surf == pytest.approx(bulk)


@pytest.mark.parametrize("angular_grid, resolution, npoints, lmax", GRID_CONFIGS)
def test_surface_reduction(angular_grid, resolution, npoints, lmax):
    """Test the projected Schur-complement traction."""
    sys = System(
        resolution,
        2.0,
        2.0,
        1e-2,
        f_ext=0.5,
        f_loc=(0.7, 1.2),
        kbt=0.0,
        angular_grid=angular_grid,
        lmax=lmax,
    )

    surf = np.random.default_rng(1).uniform(low=-1e-3, high=1e-3, size=sys.nsurf)
    bulk = slice(None, -sys.nsurf)
    surface = slice(-sys.nsurf, None)
    raw_trac_red = sys._trac[:, surface] + sys._trac[:, bulk] @ sys._bulk_from_surf

    rigid = sys._surface_rigid_modes()
    u, singular_values, _ = np.linalg.svd(rigid, full_matrices=False)
    rank = np.count_nonzero(singular_values > 1e-12 * singular_values[0])
    qrigid = u[:, :rank]
    project_elastic = np.eye(sys.nsurf) - qrigid @ qrigid.T

    state = sys._reconstruct_state(project_elastic @ surf).ravel()

    assert sys._trac_red == pytest.approx(
        project_elastic @ raw_trac_red @ project_elastic
    )
    assert sys._trac_red @ surf == pytest.approx(project_elastic @ (sys._trac @ state))
    assert sys._trac_red @ rigid == pytest.approx(0, abs=1e-10)
    assert rigid.T @ sys._trac_red == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("angular_grid, resolution, npoints, lmax", GRID_CONFIGS)
def test_rotation_projection(angular_grid, resolution, npoints, lmax):
    """Split noise into a proper rotation and a rotation-free full-body field."""
    sys = System(resolution, 2.0, 2.0, 1e-2, lmax=lmax, angular_grid=angular_grid)
    assert sys.rotation() == pytest.approx(np.eye(DIM))
    omega = np.array([0.01, -0.02, 0.03])
    rigid = sys._surface_rotation_modes @ omega
    internal, extracted = sys._split_noise(rigid)
    assert internal == pytest.approx(0, abs=1e-12)
    assert extracted == pytest.approx(omega)

    noise = np.random.default_rng(10).normal(size=sys.nsurf)
    internal, extracted = sys._split_noise(noise)
    positions = sys._reference_xyz - sys._center
    modes = np.stack([np.cross(np.eye(DIM)[a], positions) for a in range(DIM)], axis=-1)
    residual = sys._reconstruct_state(internal)
    overlap = np.einsum("p,pai,pa->i", sys._volume_weights, modes, residual)
    assert overlap == pytest.approx(0, abs=1e-10)
    assert noise == pytest.approx(internal + sys._surface_rotation_modes @ extracted)


@pytest.mark.parametrize("method", ["euler", "trotter", "strang"])
def test_rotation_steps(method):
    """Each integrator composes the fitted noise as finite SO(3) rotations."""
    start = Rotation.from_rotvec([0.2, 0.1, -0.1]).as_matrix()
    sys = System(
        (5, 12), 1.0, 1.0, 1e-2, kbt=0.4, lmax=5, method=method, rng=123, rotation=start
    )
    assert sys.rotation() == pytest.approx(start)
    copied = sys.rotation()
    copied[:] = 0
    assert sys.rotation() == pytest.approx(start)
    random = np.random.default_rng(123)
    expected = start.copy()
    for _ in range(3):
        noise = sys._std * random.normal(size=sys.nsurf)
        if method == "strang":
            noise = expm_multiply(sys._trac_oper, noise, traceA=sys._trace)
        omega = sys._rotation_fit @ noise
        expected = expected @ Rotation.from_rotvec(omega).as_matrix()
        sys()
    assert sys.rotation() == pytest.approx(expected)
    assert sys.rotation().T @ sys.rotation() == pytest.approx(np.eye(DIM), abs=1e-12)
    assert np.linalg.det(sys.rotation()) == pytest.approx(1)


@pytest.mark.parametrize("method", ["euler", "trotter", "strang"])
@pytest.mark.parametrize("angular_grid, resolution, npoints, lmax", GRID_CONFIGS)
def test_pure_rotational_noise_step(
    method, angular_grid, resolution, npoints, lmax, monkeypatch
):
    """Pure rotational noise turns the frame without deforming the body."""
    sys = System(
        resolution,
        1.0,
        1.0,
        1e-2,
        kbt=0.4,
        lmax=lmax,
        angular_grid=angular_grid,
        method=method,
    )
    omega = np.array([0.01, -0.02, 0.03])
    normal = (sys._surface_rotation_modes @ omega) / sys._std

    class FixedNoise:
        def normal(self, size):
            assert size == sys.nsurf
            return normal

    monkeypatch.setattr(sys, "_rng", FixedNoise())
    sys()
    assert sys.state() == pytest.approx(0, abs=1e-11)
    assert sys.rotation() == pytest.approx(Rotation.from_rotvec(omega).as_matrix())


def test_rotation_validation_and_lab_force():
    """Reject invalid orientations and rotate a lab-fixed force into body axes."""
    for invalid in (np.zeros((3, 3)), np.diag([-1, 1, 1]), np.full((3, 3), np.nan)):
        with pytest.raises(ValueError, match="SO\\(3\\)"):
            System((5, 12), 1.0, 1.0, 0.01, lmax=5, rotation=invalid)
    rotation = Rotation.from_rotvec([0, np.pi / 2, 0]).as_matrix()
    sys = System((5, 12), 1.0, 1.0, 0.01, lmax=5, kbt=0.0, f_ext=1.0, rotation=rotation)
    assert sys._body_force().reshape(-1, DIM) == pytest.approx(
        sys._force.reshape(-1, DIM) @ rotation
    )
    sys()
    assert sys.rotation() == pytest.approx(rotation)


def _vtk_array(root: ET.Element, name: str | None) -> np.ndarray:
    """Return values from a VTK XML DataArray."""
    for data_array in root.iter("DataArray"):
        if data_array.get("Name") == name:
            return np.fromstring(data_array.text or "", sep=" ")
    raise AssertionError(f"Missing DataArray {name!r}")


def _vtk_points(root: ET.Element) -> np.ndarray:
    """Return points from a VTK XML file."""
    data_array = root.find(".//Points/DataArray")
    assert data_array is not None
    return np.fromstring(data_array.text or "", sep=" ").reshape(-1, DIM)


def _surface_points_cartesian(sys: System) -> np.ndarray:
    """Return Cartesian coordinates of system surface points."""
    radius, theta, phi = sys.surface_points().T
    return np.column_stack(
        (
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta),
        )
    )


@pytest.mark.parametrize("angular_grid, resolution, npoints, lmax", GRID_CONFIGS)
def test_export_vts(tmp_path, angular_grid, resolution, npoints, lmax):
    """Test exporting surface data as a VTK StructuredGrid."""
    state = np.random.default_rng(0).uniform(low=-1e-3, high=1e-3, size=(npoints, DIM))
    rotation = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_matrix()
    sys = System(
        resolution,
        1.0,
        1.0,
        1e-2,
        kbt=0.0,
        state=state,
        angular_grid=angular_grid,
        lmax=lmax,
        rotation=rotation,
    )

    dense_resolution = 7
    assert sys.export_vts(tmp_path, resolution=dense_resolution) is sys
    filepath = tmp_path / "surface.vts"
    assert filepath.exists()

    root = ET.parse(filepath).getroot()
    assert root.get("type") == "StructuredGrid"
    piece = root.find(".//Piece")
    assert piece is not None
    assert piece.get("Extent") == (
        f"0 {dense_resolution - 1} 0 {dense_resolution - 1} 0 0"
    )

    points = _vtk_points(root)
    displacement = _vtk_array(root, "Displacement_Vector").reshape(-1, DIM)
    magnitude = _vtk_array(root, "Displacement_Magnitude")

    assert points.shape == (dense_resolution * dense_resolution, DIM)
    assert displacement.shape == points.shape
    assert magnitude.shape == (len(points),)
    assert magnitude == pytest.approx(np.linalg.norm(displacement, axis=1))
    assert np.isfinite(points).all()
    theta = np.linspace(0, np.pi, dense_resolution)
    phi = np.linspace(0, 2 * np.pi, dense_resolution)
    tg, pg = np.meshgrid(theta, phi)
    radius = np.mean(sys.surface_points()[:, 0])
    reference = sys._spherical_to_cartesian(
        np.column_stack((np.full(len(tg.ravel()), radius), tg.ravel(), pg.ravel()))
    )
    internal = sys._build_vts_projection(dense_resolution) @ sys.surface()
    expected_points = sys._center + (reference + internal - sys._center) @ rotation.T
    assert points == pytest.approx(expected_points)
    assert displacement == pytest.approx(points - reference)


@pytest.mark.parametrize("angular_grid, resolution, npoints, lmax", GRID_CONFIGS)
def test_export_vts_projection_cache(tmp_path, angular_grid, resolution, npoints, lmax):
    """Test that VTS export can read a cached projection matrix."""
    state = np.random.default_rng(0).uniform(low=-1e-3, high=1e-3, size=(npoints, DIM))
    sys = System(
        resolution,
        1.0,
        1.0,
        1e-2,
        kbt=0.0,
        state=state,
        angular_grid=angular_grid,
        lmax=lmax,
    )

    dense_resolution = 7
    cache = tmp_path / "projection.h5"
    uncached = tmp_path / "uncached.vts"
    cached = tmp_path / "cached.vts"

    sys.save_projection_cache(cache, resolution=dense_resolution)
    sys.export_vts(uncached, resolution=dense_resolution)
    assert (
        sys.export_vts(
            cached,
            resolution=dense_resolution,
            projection_cache=cache,
        )
        is sys
    )

    uncached_root = ET.parse(uncached).getroot()
    cached_root = ET.parse(cached).getroot()

    assert _vtk_points(cached_root) == pytest.approx(_vtk_points(uncached_root))
    assert _vtk_array(cached_root, "Displacement_Vector") == pytest.approx(
        _vtk_array(uncached_root, "Displacement_Vector")
    )
    assert _vtk_array(cached_root, "Displacement_Magnitude") == pytest.approx(
        _vtk_array(uncached_root, "Displacement_Magnitude")
    )


@pytest.mark.parametrize("angular_grid, resolution, npoints, lmax", GRID_CONFIGS)
def test_export_vtp(tmp_path, angular_grid, resolution, npoints, lmax):
    """Test export of the surface data as VTK PolyData."""
    state = np.random.default_rng(0).uniform(low=-1e-3, high=1e-3, size=(npoints, DIM))
    rotation = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_matrix()
    sys = System(
        resolution,
        1.0,
        1.0,
        1e-2,
        kbt=0.0,
        state=state,
        angular_grid=angular_grid,
        lmax=lmax,
        rotation=rotation,
    )

    filepath = tmp_path / "native.vtp"
    assert sys.export_vtp(filepath) is sys
    assert filepath.exists()

    root = ET.parse(filepath).getroot()
    assert root.get("type") == "PolyData"
    piece = root.find(".//Piece")
    assert piece is not None
    assert int(piece.get("NumberOfPoints", "0")) == sys.mesh[1]

    points = _vtk_points(root)
    displacement = _vtk_array(root, "Displacement_Vector").reshape(-1, DIM)
    magnitude = _vtk_array(root, "Displacement_Magnitude")
    point_ids = _vtk_array(root, "Surface_Point_Id").astype(int)
    connectivity = _vtk_array(root, "connectivity").astype(int)
    offsets = _vtk_array(root, "offsets").astype(int)

    reference = _surface_points_cartesian(sys)
    expected = sys._center + (reference + sys.surface() - sys._center) @ rotation.T
    assert points == pytest.approx(expected)
    assert displacement == pytest.approx(points - reference)
    assert magnitude == pytest.approx(np.linalg.norm(displacement, axis=1))
    assert np.array_equal(point_ids, np.arange(sys.mesh[1]))
    assert connectivity.min() >= 0
    assert connectivity.max() < sys.mesh[1]

    cell_size = 3 if angular_grid == "fibonacci" else 4
    assert np.all(np.diff(np.r_[0, offsets]) == cell_size)
    assert int(piece.get("NumberOfPolys", "0")) == len(offsets)
