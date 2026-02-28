"""Tests for `hookeanity.system` module."""

import numpy as np
from hookeanity.base import DIM
from hookeanity.system import System


def test_system_init():
    """Test `hookeanity.System` initialization."""
    mesh = (5, 25)
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
    mesh = (5, 25)
    system = System(mesh, lam, mu, f_ext, kbt)
    system.save(filepath)
    sample1 = system._rng.random(5)

    loaded = System.load(filepath)
    sample2 = loaded._rng.random(5)
    assert loaded.mesh == mesh
    assert loaded._lam == lam
    assert loaded._mu == mu
    assert loaded._ext == f_ext
    assert loaded._kbt == kbt
    assert np.all(sample1 == sample2)


def test_ball_system():
    """Test unit ball without forces."""
    mesh = (5, 25)
    dt = 1e-2
    sys = System(mesh, 1.0, 1.0, kbt=0.0)
    assert sys._kbt == 0.0
    assert sys(dt)._state.shape == (np.prod(mesh), DIM)
    assert np.all(sys._state == 0)
