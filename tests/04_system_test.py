"""Tests for `hookeanity.system` module"""

import pytest
import numpy as np
from hookeanity.system import System

def test_initialization():
    """Test initialization"""
    lam = 1.0
    mu = 1.0
    f_ext = 0.0
    order = (2, 2)
    state = np.zeros((3, (order[0] + 1) * order[1]))
    sys = System(
        state=state,
        lam=lam,
        mu=mu,
        f_ext=f_ext,
        order=order,
    )

    # print('\nbfs:', sys._inter.bfs.shape)
    # print(len(list(sys.points())))
    # print(sys.npoints)
    # w, _, _ = sys._inter.weights
    # print(len(w))
    # print(sys._radial_surface.shape)
    # # print(list(sys.points()))
    # # print(np.array([p[0] for p in sys.points()]))
    # print(sys._radial_surface.shape, sys._radial_on_points.shape) # has to be extended to match point shape

    assert sys._state.shape == state.shape
    assert sys._lam == lam
    assert sys._mu == mu
    assert sys._f_ext == f_ext
    assert sys._radial_surface.shape == (order[0] + 1,)
    assert sys._radial_on_points.shape == (order[0] + 1, sys.npoints)
    assert sys._inverse_lhs.shape[0] == sys._inverse_lhs.shape[1]

def test_dummy():
    """Test shapes after one step"""
    lam = 1.0
    mu = 1.0
    f_ext = 0.0
    order = (2, 2)
    state = np.zeros((3, (order[0] + 1) * order[1]))
    dt = 0.1
    sys = System(state=state, lam=lam, mu=mu, f_ext=f_ext, order=order)

    assert sys._state.shape[1] == sys.npoints
    assert sys._elastic_stress() == pytest.approx(state)
    assert sys._external_force() == pytest.approx(np.zeros(sys.npoints))
    assert sys._fluctuations(dt).shape == (3, sys.npoints)
    assert sys(dt)._state.shape == pytest.approx(state.shape)