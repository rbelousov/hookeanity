"""Tests for `hookeanity.system` module."""

import pytest
import numpy as np
from hookeanity.base import DIM
from hookeanity.system import System


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

    transform = sys._inter.transform()
    inverse = sys._inter.inverse()
    partial = np.asarray(
        [sys._inter.partial_x(), sys._inter.partial_y(), sys._inter.partial_z()]
    )
    noc = sys._state - sys._state.mean(axis=0)

    grad_div = np.einsum("lL,aLM,bMN,Nn->lanb", inverse, partial, partial, transform)
    laplacian = np.einsum("lL,bLM,bMN,Nn->ln", inverse, partial, partial, transform)
    expected = (
        (sys._mu + sys._lam) * np.einsum("lanb,nb->la", grad_div, noc)
        + sys._mu * np.einsum("ln,na->la", laplacian, noc)
    ).ravel()

    vec = noc.ravel()
    sz = len(vec)
    gd = np.einsum("lL,aLM,bMN,Nn->lanb", inverse, partial, partial, transform).reshape(
        sz, sz
    )
    lap = np.einsum(
        "ab,lL,cLM,cMN,Nn->lanb", np.eye(DIM), inverse, partial, partial, transform
    ).reshape(sz, sz)
    mtx = (sys._mu + sys._lam) * gd + sys._mu * lap
    value = mtx @ vec
    assert value == pytest.approx(expected)

    surf = DIM * mesh[1]
    value = mtx[:-surf, :] @ vec
    assert value == pytest.approx(expected[:-surf])


def test_traction():
    """Test formulas for the traction force."""
    mesh = (5, 6)
    state = np.random.default_rng(0).uniform(
        low=-1e-3, high=1e-3, size=(np.prod(mesh), DIM)
    )
    sys = System(mesh, 1.0, 1.0, kbt=0.0, state=state)

    eye = np.eye(DIM)
    npoints = sys.npoints
    noc = sys._state - sys._state.mean(axis=0)
    prj = np.kron(np.eye(npoints) - np.ones((npoints, npoints)) / npoints, eye)
    assert prj @ sys._state.ravel() == pytest.approx(noc.ravel())

    transform = sys._inter.transform()
    inverse = sys._inter.inverse()
    partial_xyz = np.array(
        [sys._inter.partial_x(), sys._inter.partial_y(), sys._inter.partial_z()]
    )
    partial = np.einsum("mM,aMN,Nn->man", inverse, partial_xyz, transform)

    normals = sys.normals()
    nabla = np.einsum("pan,nb->pab", partial[-mesh[1] :, :, :], noc)
    div = np.trace(nabla, axis1=1, axis2=2)
    sym = nabla + np.swapaxes(nabla, 1, 2)
    stress = sys._lam * div[:, None, None] * eye[None, :, :] + sys._mu * sym
    expected = np.einsum("pa,pab->pb", normals, stress).ravel()

    term0 = np.einsum("pa,pan->pn", normals, partial[-mesh[1] :, :, :])
    term1 = sys._mu * np.einsum("pn,bc->pbnc", term0, eye)
    term2 = sys._lam * np.einsum("pb,pcn->pbnc", normals, partial[-mesh[1] :, :, :])
    term3 = sys._mu * np.einsum("pc,pbn->pbnc", normals, partial[-mesh[1] :, :, :])
    trac = (term1 + term2 + term3).reshape(DIM * mesh[1], DIM * np.prod(mesh))
    value = trac @ noc.ravel()
    assert value == pytest.approx(expected)

    value = sys._trac @ sys._state.ravel()
    assert value == pytest.approx(expected)


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
