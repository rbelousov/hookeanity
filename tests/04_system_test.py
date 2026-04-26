"""Tests for `hookeanity.system` module."""

import pytest
import numpy as np
from hookeanity.base import DIM
from hookeanity.system import System
from hookeanity.interpolation import Interpolation


def test_system_init():
    """Test `hookeanity.System` initialization."""
    mesh = (5, 12)
    sys = System(mesh, 1.0, 1.0, 1e-2, lmax=5)
    assert sys._state.shape == (np.prod(mesh), DIM)
    assert np.all(sys._state == 0)


def test_save_and_load(tmp_path):
    """Test saving and loading system data in the HDF5 format."""
    filepath = tmp_path / "system.h5"

    lam = 2.0
    mu = 3.0
    dt = 1e-2
    gamma = 2.5
    f_ext = 0.5
    f_loc = (0.7, 1.2)
    kbt = 1.5
    lmax = 5
    mesh = (5, 12)
    state = np.random.default_rng(0).uniform(
        low=-1e-3, high=1e-3, size=(np.prod(mesh), DIM)
    )
    method = "trotter"
    system = System(
        mesh,
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
    )
    system.save(filepath)
    sample1 = system._rng.random(5)
    assert system.mesh == mesh
    assert system.lmax == lmax
    assert system.surface().shape == (mesh[1], DIM)
    assert system.surface_points().shape == (mesh[1], DIM)

    loaded = System.load(filepath)
    sample2 = loaded._rng.random(5)
    assert loaded.mesh == mesh
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


def test_stress_divergence():
    """Test formulas for the divergence of stress."""
    mesh = (5, 12)
    lmax = 5
    state = np.random.default_rng(0).uniform(
        low=-1e-3, high=1e-3, size=(np.prod(mesh), DIM)
    )
    sys = System(mesh, 1.0, 1.0, 1e-2, kbt=0.0, state=state, lmax=lmax)

    inter = Interpolation(mesh, lmax=lmax)
    transform = inter.transform()
    inverse = inter.inverse()
    partial = np.asarray([inter.partial_x(), inter.partial_y(), inter.partial_z()])

    grad_div = np.einsum("lL,aLM,bMN,Nn->lanb", inverse, partial, partial, transform)
    laplacian = np.einsum("lL,bLM,bMN,Nn->ln", inverse, partial, partial, transform)
    expected = (
        (sys._mu + sys._lam) * np.einsum("lanb,nb->la", grad_div, sys._state)
        + sys._mu * np.einsum("ln,na->la", laplacian, sys._state)
    ).ravel()

    ndof = sys.ndof
    gd = np.einsum("lL,aLM,bMN,Nn->lanb", inverse, partial, partial, transform).reshape(
        ndof, ndof
    )
    lap = np.einsum(
        "ab,lL,cLM,cMN,Nn->lanb", np.eye(DIM), inverse, partial, partial, transform
    ).reshape(ndof, ndof)
    mtx = (sys._mu + sys._lam) * gd + sys._mu * lap
    value = mtx @ sys._state.ravel()
    assert value == pytest.approx(expected)

    value = sys._ela @ sys._state.ravel()
    assert value == pytest.approx(expected[: -DIM * mesh[1]])


def test_traction():
    """Test formulas for the traction force."""
    mesh = (5, 12)
    lmax = 5
    state = np.random.default_rng(0).uniform(
        low=-1e-3, high=1e-3, size=(np.prod(mesh), DIM)
    )
    sys = System(mesh, 1.0, 1.0, 1e-2, kbt=0.0, state=state, lmax=lmax)

    inter = Interpolation(mesh, lmax=lmax)
    eye = np.eye(DIM)
    transform = inter.transform()
    inverse = inter.inverse()
    partial_xyz = np.array([inter.partial_x(), inter.partial_y(), inter.partial_z()])
    partial = np.einsum("mM,aMN,Nn->man", inverse, partial_xyz, transform)

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


def test_system_smoke():
    """Check first smoke of the system functionality."""
    mesh = (5, 12)
    lmax = 5
    dt = 1e-2

    sys = System(mesh, 1.0, 1.0, dt, kbt=0.0, lmax=lmax)
    assert sys._kbt == 0.0
    assert sys()._state.shape == (np.prod(mesh), DIM)
    assert np.all(sys._state == 0)

    sys._state += 1
    assert np.all(sys._state == 1)
    assert sys()._state.shape == (np.prod(mesh), DIM)
    assert sys._state == pytest.approx(1, abs=1e-3)

    sys = System(mesh, 1.0, 1.0, dt, verbose=True, lmax=lmax)()
    assert sys._kbt == 1
    assert sys._state != pytest.approx(1, abs=1e-3)

    sys_n = System(mesh, 1.0, 1.0, dt, f_ext=10, f_loc=(0, 0), kbt=0.0, lmax=lmax)
    sys_s = System(mesh, 1.0, 1.0, dt, f_ext=10, f_loc=(np.pi, 0), kbt=0.0, lmax=lmax)
    assert not (sys_n._force == pytest.approx(sys_s._force))
    sys_n()()()
    sys_s()()()
    assert not (sys_n._state == pytest.approx(sys_s._state))

    sys_trotter = System(
        mesh,
        1.0,
        1.0,
        dt,
        f_ext=10,
        f_loc=(np.pi, 0),
        kbt=0.0,
        method="trotter",
        lmax=lmax,
    )
    sys_trotter()
    assert np.isfinite(sys_trotter._state).all()

    sys_strang = System(
        mesh, 1.0, 1.0, dt, f_ext=10, f_loc=(0, 0), kbt=0.0, method="strang", lmax=lmax
    )
    sys_strang()
    assert np.isfinite(sys_strang._state).all()


def test_system_cache(tmp_path):
    """Test saving and loading cache data for `System` objects."""
    mesh = (5, 12)
    lmax = 5
    cache = tmp_path / "cache.5x6.l2m3.h5"
    sys1 = System(mesh, 2.0, 3.0, 1e-2, lmax=lmax)
    sys1.save_cache(cache)

    sys2 = System(mesh, 2.0, 3.0, 1e-2, lmax=lmax, cache=cache)
    assert sys2.lmax == sys1.lmax
    assert np.all(sys1._grid == sys2._grid)
    assert np.all(sys1._trac == sys2._trac)
    assert np.all(sys1._ela == sys2._ela)
    assert np.all(sys1._bulk_from_surf == sys2._bulk_from_surf)
    assert np.all(sys1._trac_red == sys2._trac_red)
    assert np.all(sys1._force == sys2._force)

    with pytest.raises(AssertionError, match="spherical-harmonic cutoffs"):
        System(mesh, 2.0, 3.0, 1e-2, cache=cache)

    with pytest.raises(AssertionError, match="time steps"):
        System(mesh, 2.0, 3.0, 2e-2, lmax=lmax, cache=cache)


def _build_operators_old(
    mesh: tuple[int, int],
    lam: float,
    mu: float,
    f_ext: float,
    f_loc: tuple[float, float],
    lmax: int,
):
    """Implement `_build_operators` according to the old method.

    Returns:
        Tuple `(grid, trac, ela, force)`.
    """
    ndof = DIM * int(np.prod(mesh))
    eye = np.eye(DIM)

    inter = Interpolation(mesh, lmax=lmax)
    grid = inter.grid()

    # Derivative operators
    transform = inter.transform()
    inverse = inter.inverse()
    partial_xyz = np.array([inter.partial_x(), inter.partial_y(), inter.partial_z()])
    partial = np.einsum("mM,aMN,Nn->man", inverse, partial_xyz, transform)

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
    ).reshape(nbulk, ndof)

    vl = np.einsum(
        "ab,lL,cLM,cMN,Nn->lanb",
        eye,
        inverse[: -mesh[1]],
        partial_xyz,
        partial_xyz,
        transform,
    ).reshape(nbulk, ndof)

    ela = (mu + lam) * gd + mu * vl

    # Additive forces
    force = np.zeros((mesh[1], DIM))
    force[:, 2] = f_ext * np.array(
        [chi(np.array(f_loc)) for chi in inter._angular_functions]
    )
    force = force.ravel()

    return trac, ela, force


def test_build_operators():
    """Test that the optimized `_build_operators()` matches the old implementation."""
    mesh = (5, 12)
    lmax = 5
    lam = 2.0
    mu = 3.0
    dt = 1e-2
    f_ext = 0.5
    f_loc = (0.7, 1.2)

    sys = System(mesh, lam, mu, dt, f_ext=f_ext, f_loc=f_loc, kbt=0.0, lmax=lmax)
    trac_old, ela_old, force_old = _build_operators_old(
        mesh=mesh, lam=lam, mu=mu, f_ext=f_ext, f_loc=f_loc, lmax=lmax
    )

    assert sys._trac.shape == trac_old.shape
    assert sys._ela.shape == ela_old.shape
    assert sys._force.shape == force_old.shape

    assert sys._trac == pytest.approx(trac_old)
    assert sys._ela == pytest.approx(ela_old)
    assert sys._force == pytest.approx(force_old)


def test_bulk_reconstruction():
    """Test the reduced bulk map reproduces the direct solve from Eq."""
    mesh = (5, 12)
    lmax = 5
    sys = System(mesh, 2.0, 3.0, 1e-2, f_ext=0.5, f_loc=(0.7, 1.2), kbt=0.0, lmax=lmax)

    surf = np.random.default_rng(0).uniform(low=-1e-3, high=1e-3, size=sys.nsurf)
    bulk = -np.linalg.solve(
        sys._ela[:, : -sys.nsurf],
        sys._ela[:, -sys.nsurf :] @ surf,
    )

    assert sys._bulk_from_surf @ surf == pytest.approx(bulk)


def test_surface_reduction():
    """Test the Schur-complement traction against the full traction operator."""
    mesh = (5, 12)
    lmax = 5
    sys = System(mesh, 2.0, 3.0, 1e-2, f_ext=0.5, f_loc=(0.7, 1.2), kbt=0.0, lmax=lmax)

    surf = np.random.default_rng(1).uniform(low=-1e-3, high=1e-3, size=sys.nsurf)
    state = sys._reconstruct_state(surf).ravel()

    assert sys._trac_red @ surf == pytest.approx(sys._trac @ state)
