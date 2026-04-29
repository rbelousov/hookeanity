"""Tests for `hookeanity.system` module."""

import pytest
import numpy as np
from typing import cast
from hookeanity.base import DIM
from hookeanity.system import System, Init
from hookeanity.interpolation import Interpolation


GRID_CONFIGS = [
    ("fibonacci", (5, 12), 60, 5),
    ("gauss-legendre-fourier", (5, 5), 250, 15),
]


@pytest.fixture(
    scope="module",
    params=[
        ("fibonacci", (5, 12), 60, 5),
        ("gauss-legendre-fourier", (4, 5), 200, 15),
    ],
)
def system(request):
    """Return System object for different grids, with a random initial state."""
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

    lam, mu, dt = 2.0, 2.0, 1e-2
    gamma, f_ext = 2.5, 0.5
    f_loc = (0.7, 1.2)
    kbt = 1.5
    method = "trotter"

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
    assert np.all(sample1 == sample2)


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
    assert not (sys_n._force == pytest.approx(sys_s._force))
    sys_n()()()
    sys_s()()()
    assert not (sys_n._state == pytest.approx(sys_s._state))

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
        init="no_translations",
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
        init="no_translations",
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
            init="no_translations",
        )


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
    """Implement `_build_operators` according to the old method.

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
    """Test that the optimized `_build_operators()` matches the old implementation."""
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
    """Test the reduced bulk map reproduces the direct solve from Eq."""
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


@pytest.mark.parametrize(
    ("init", "left_mode", "right_mode"),
    [
        ("no_rigid_motion", "no_rigid_motion", "no_rigid_motion"),
        ("no_translations", "no_translations", "no_translations"),
        ("no_rotations", "no_rotations", "no_rotations"),
        (None, None, None),
    ],
)
def test_surface_reduction_init_modes(
    init: str | None, left_mode: str | None, right_mode: str | None
):
    """Test traction reduction under configurable rigid-motion projections."""
    sys = System(
        (5, 12), 2.0, 3.0, 1e-2, f_ext=0.0, kbt=0.0, lmax=5, init=cast(Init, init)
    )
    bulk = slice(None, -sys.nsurf)
    surface = slice(-sys.nsurf, None)
    raw_trac_red = sys._trac[:, surface] + sys._trac[:, bulk] @ sys._bulk_from_surf

    left_modes = sys._surface_rigid_modes(cast(Init, left_mode))
    right_modes = sys._surface_rigid_modes(cast(Init, right_mode))

    def projector(modes: np.ndarray) -> np.ndarray:
        if modes.shape[1] == 0:
            return np.eye(sys.nsurf)
        u, singular_values, _ = np.linalg.svd(modes, full_matrices=False)
        rank = np.count_nonzero(singular_values > 1e-12 * singular_values[0])
        return np.eye(sys.nsurf) - u[:, :rank] @ u[:, :rank].T

    pl = projector(left_modes)
    pr = projector(right_modes)
    expected = pl @ raw_trac_red @ pr
    assert sys._trac_red == pytest.approx(expected)

    if left_modes.shape[1]:
        assert left_modes.T @ sys._trac_red == pytest.approx(0, abs=1e-10)
    if right_modes.shape[1]:
        assert sys._trac_red @ right_modes == pytest.approx(0, abs=1e-10)
