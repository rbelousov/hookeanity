"""Tests for `hookeanity.system` module."""

import pytest
import numpy as np
from hookeanity.base import DIM
from hookeanity.system import System
from hookeanity.interpolation import Interpolation


def test_system_init():
    """Test `hookeanity.System` initialization."""
    mesh = (5, 6)
    sys = System(mesh, 1.0, 1.0)
    assert sys._state.shape == (np.prod(mesh), DIM)
    assert np.all(sys._state == 0)


def test_save_and_load(tmp_path):
    """Test saving and loading system data in the HDF5 format."""
    filepath = tmp_path / "system.h5"

    lam = 2.0
    mu = 3.0
    f_ext = 0.5
    kbt = 1.5
    mesh = (5, 6)
    system = System(mesh, lam, mu, f_ext, kbt=kbt)
    system.save(filepath)
    sample1 = system._rng.random(5)
    assert system.mesh == mesh
    assert system.surface().shape == (mesh[1], DIM)
    assert system.surface_points().shape == (mesh[1], DIM)

    loaded = System.load(filepath)
    sample2 = loaded._rng.random(5)
    assert loaded.mesh == mesh
    assert loaded._lam == lam
    assert loaded._mu == mu
    assert loaded._ext == f_ext
    assert loaded._kbt == kbt
    assert np.all(sample1 == sample2)


def test_stress_divergence():
    """Test formulas for the divergence of stress."""
    mesh = (5, 6)
    state = np.random.default_rng(0).uniform(
        low=-1e-3, high=1e-3, size=(np.prod(mesh), DIM)
    )
    sys = System(mesh, 1.0, 1.0, kbt=0.0, state=state)

    inter = Interpolation(mesh)
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

    sys.apply_rigid_motion_projection()
    value = sys._ela @ sys._state.ravel()
    assert value == pytest.approx(expected[: -DIM * mesh[1]], abs=1e-4)


def test_traction():
    """Test formulas for the traction force."""
    mesh = (5, 6)
    state = np.random.default_rng(0).uniform(
        low=-1e-3, high=1e-3, size=(np.prod(mesh), DIM)
    )
    sys = System(mesh, 1.0, 1.0, kbt=0.0, state=state)

    inter = Interpolation(mesh)
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

    sys.apply_rigid_motion_projection()
    value = sys._trac @ sys._state.ravel()
    assert value == pytest.approx(expected, abs=1e-4)


def test_system_smoke():
    """Check first smoke of the system functionality."""
    mesh = (5, 10)
    dt = 1e-2

    sys = System(mesh, 1.0, 1.0, kbt=0.0)
    assert sys._kbt == 0.0
    assert sys(dt)._state.shape == (np.prod(mesh), DIM)
    assert np.all(sys._state == 0)

    sys._state += 1
    assert np.all(sys._state == 1)
    assert sys(dt)._state.shape == (np.prod(mesh), DIM)
    assert sys._state == pytest.approx(1, abs=1e-3)

    sys = System(mesh, 1.0, 1.0, verbose=True)(dt)
    assert sys._kbt == 1
    assert sys._state != pytest.approx(1, abs=1e-3)

    sys_n = System(mesh, 1.0, 1.0, f_ext=10, f_loc=(0, 0), kbt=0.0)
    sys_s = System(mesh, 1.0, 1.0, f_ext=10, f_loc=(np.pi, 0), kbt=0.0)
    assert not (sys_n._force == pytest.approx(sys_s._force))
    sys_n(dt)(dt)(dt)
    sys_s(dt)(dt)(dt)
    assert not (sys_n._state == pytest.approx(sys_s._state))


def test_system_cache(tmp_path):
    """Test saving and loading cache data for `System` objects."""
    mesh = (5, 6)
    cache = tmp_path / "cache.5x6.l2m3.h5"
    sys1 = System(mesh, 2.0, 3.0)
    sys1.save_cache(cache)

    sys2 = System(mesh, 2.0, 3.0, cache=cache)
    assert np.all(sys1._grid == sys2._grid)
    assert np.all(sys1._trac == sys2._trac)
    assert np.all(sys1._ela == sys2._ela)
    assert np.all(sys1._force == sys2._force)
