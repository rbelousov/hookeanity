"""Tests for `hookeanity.interpolation` module."""

import pytest
import numpy as np
import scipy as sp
from hookeanity.base import DIM, ScalarField, instance, Real, Point, Grid, ScalarValues
from hookeanity.interpolation import Interpolation


@pytest.fixture
def scalar() -> ScalarField:
    """Return a test scalar field."""

    def wrapper(arg: Point | Grid) -> Real | ScalarValues:
        r = arg[0] if instance(arg, Point) else arg[:, 0]
        theta = arg[1] if instance(arg, Point) else arg[:, 1]
        phi = arg[2] if instance(arg, Point) else arg[:, 2]
        return sp.special.j0(r * np.cos(theta) * np.sin(phi))

    return wrapper


@pytest.fixture
def scalar_r() -> ScalarField:
    """Return radial derivative of the test scalar field."""

    def wrapper(arg: Point | Grid) -> Real | ScalarValues:
        r = arg[0] if instance(arg, Point) else arg[:, 0]
        theta = arg[1] if instance(arg, Point) else arg[:, 1]
        phi = arg[2] if instance(arg, Point) else arg[:, 2]

        angular_factor = np.cos(theta) * np.sin(phi)
        return -sp.special.j1(r * angular_factor) * angular_factor

    return wrapper


@pytest.fixture
def scalar_theta() -> ScalarField:
    """Return polar derivative of the test scalar field."""

    def wrapper(arg: Point | Grid) -> Real | ScalarValues:
        r = arg[0] if instance(arg, Point) else arg[:, 0]
        theta = arg[1] if instance(arg, Point) else arg[:, 1]
        phi = arg[2] if instance(arg, Point) else arg[:, 2]

        return (
            sp.special.j1(r * np.cos(theta) * np.sin(phi)) * np.sin(theta) * np.sin(phi)
        )

    return wrapper


@pytest.fixture
def scalar_phi() -> ScalarField:
    """Return azimuthal derivative of the test scalar field."""

    def wrapper(arg: Point | Grid) -> Real | ScalarValues:
        r = arg[0] if instance(arg, Point) else arg[:, 0]
        theta = arg[1] if instance(arg, Point) else arg[:, 1]
        phi = arg[2] if instance(arg, Point) else arg[:, 2]

        return (
            -sp.special.j1(r * np.cos(theta) * np.sin(phi))
            * np.cos(theta)
            * np.cos(phi)
        )

    return wrapper


@pytest.fixture
def scalar_x(scalar_r, scalar_theta, scalar_phi) -> ScalarField:
    """Return the derivative of a test scalar field with respect to the x component."""

    def wrapper(arg: Point | Grid) -> Real | ScalarValues:
        r = arg[0] if instance(arg, Point) else arg[:, 0]
        theta = arg[1] if instance(arg, Point) else arg[:, 1]
        phi = arg[2] if instance(arg, Point) else arg[:, 2]

        r_x = np.sin(theta) * np.cos(phi)
        theta_x = np.cos(theta) * np.cos(phi) / r
        phi_x = -np.sin(phi) / (r * np.sin(theta))
        return (
            r_x * scalar_r(arg) + theta_x * scalar_theta(arg) + phi_x * scalar_phi(arg)
        )

    return wrapper


@pytest.fixture
def scalar_y(scalar_r, scalar_theta, scalar_phi) -> ScalarField:
    """Return the derivative of a test scalar field with respect to the y component."""

    def wrapper(arg: Point | Grid) -> Real | ScalarValues:
        r = arg[0] if instance(arg, Point) else arg[:, 0]
        theta = arg[1] if instance(arg, Point) else arg[:, 1]
        phi = arg[2] if instance(arg, Point) else arg[:, 2]

        r_y = np.sin(theta) * np.sin(phi)
        theta_y = np.cos(theta) * np.sin(phi) / r
        phi_y = np.cos(phi) / (r * np.sin(theta))
        return (
            r_y * scalar_r(arg) + theta_y * scalar_theta(arg) + phi_y * scalar_phi(arg)
        )

    return wrapper


@pytest.fixture
def scalar_z(scalar_r, scalar_theta, scalar_phi) -> ScalarField:
    """Return the derivative of a test scalar field with respect to the z component."""

    def wrapper(arg: Point | Grid) -> Real | ScalarValues:
        r = arg[0] if instance(arg, Point) else arg[:, 0]
        theta = arg[1] if instance(arg, Point) else arg[:, 1]
        # phi = arg[2] if instance(arg, Point) else arg[:, 2]

        r_z = np.cos(theta)
        theta_z = -np.sin(theta) / r
        phi_z = 0.0
        return (
            r_z * scalar_r(arg) + theta_z * scalar_theta(arg) + phi_z * scalar_phi(arg)
        )

    return wrapper


def random_points(n: int, sz: int) -> np.ndarray:
    """Sample n random indices up to sz."""
    rng = np.random.default_rng(seed=0)
    return rng.choice(range(sz), size=n, replace=False)


def test_transform():
    """Test `Interpolation.transform()` and `Interpolation.inverse()`"""
    inter = Interpolation((5, 5))
    sz = len(inter)
    shape = (sz, sz)
    transform = inter.transform()
    inverse = inter.inverse()
    assert transform.shape == shape
    assert inverse.shape == shape
    assert inverse @ transform == pytest.approx(np.eye(sz))


def test_partial_r(scalar, scalar_r):
    """Test `Interpolation.partial_r`"""
    inter = Interpolation((5, 5))
    grid = inter.grid()
    values = scalar(grid)
    modes = inter.transform() @ values
    modes_r = inter.partial_r() @ modes

    indices = random_points(10, len(inter))
    points = grid[indices]
    values_r = inter.inverse()[indices] @ modes_r
    expected = scalar_r(points)

    assert values_r == pytest.approx(expected, abs=1e-3)


def test_partial_theta(scalar, scalar_theta):
    """Test `Interpolation.partial_theta`"""
    inter = Interpolation((12, 144))
    grid = inter.grid()

    values = scalar(grid)
    modes = inter.transform() @ values
    modes_theta = inter.partial_theta() @ modes

    indices = random_points(10, len(inter))
    points = grid[indices]
    values_theta = inter.inverse()[indices] @ modes_theta
    expected = scalar_theta(points)
    assert values_theta == pytest.approx(expected, abs=1e-1)


def test_partial_phi(scalar, scalar_phi):
    """Test `Interpolation.partial_phi`"""
    inter = Interpolation((12, 144))
    grid = inter.grid()

    values = scalar(grid)
    modes = inter.transform() @ values
    modes_phi = inter.partial_phi() @ modes

    rng = np.random.default_rng(seed=0)
    indices = rng.choice(range(len(inter)), size=10, replace=False)
    points = grid[indices]
    values_phi = inter.inverse()[indices] @ modes_phi
    expected = scalar_phi(points)
    assert values_phi == pytest.approx(expected, abs=1e-1)


def test_jacobian():
    """Test elements of the jacobian matrix."""
    inter = Interpolation((10, 10))
    sz = len(inter)

    jx = inter._jacobian_x(inter.grid())
    assert jx.shape == (sz, DIM)
    assert np.all(np.isfinite(jx))

    jy = inter._jacobian_y(inter.grid())
    assert jy.shape == (sz, DIM)
    assert np.all(np.isfinite(jy))

    jz = inter._jacobian_z(inter.grid())
    assert jz.shape == (sz, DIM)
    assert np.all(np.isfinite(jz))


def test_x_identity():
    """Test `Interpolation.partial_x` in a simple setting."""

    def x_identity(arg: Point | Grid) -> Real | ScalarValues:
        r = arg[0] if instance(arg, Point) else arg[:, 0]
        theta = arg[1] if instance(arg, Point) else arg[:, 1]
        phi = arg[2] if instance(arg, Point) else arg[:, 2]
        return r * np.sin(theta) * np.cos(phi)

    inter = Interpolation((10, 10))
    grid = inter.grid()
    cfs = inter.transform() @ x_identity(grid)

    indices = random_points(10, len(inter))
    precision = 1e-1

    oper_x = inter.partial_x()
    cfs_x = np.dot(oper_x, cfs)
    values = inter.inverse()[indices] @ cfs_x
    expected = np.array([1.0] * len(indices))
    assert values == pytest.approx(expected, rel=precision)

    oper_xy = np.dot(oper_x, inter.partial_y())
    cfs_xy = np.dot(oper_xy, cfs)
    values = inter.inverse()[indices] @ cfs_xy
    expected = np.array([0.0] * len(indices))
    assert values == pytest.approx(expected, abs=precision)

    oper_xz = np.dot(oper_x, inter.partial_z())
    cfs_xz = np.dot(oper_xz, cfs)
    values = inter.inverse()[indices] @ cfs_xz
    expected = np.array([0.0] * len(indices))
    assert values == pytest.approx(expected, abs=precision)


def test_y_identity():
    """Test `Interpolation.partial_y` in a simple setting."""

    def y_identity(arg: Point | Grid) -> Real | ScalarValues:
        r = arg[0] if instance(arg, Point) else arg[:, 0]
        theta = arg[1] if instance(arg, Point) else arg[:, 1]
        phi = arg[2] if instance(arg, Point) else arg[:, 2]
        return r * np.sin(theta) * np.sin(phi)

    inter = Interpolation((10, 10))
    grid = inter.grid()
    cfs = inter.transform() @ y_identity(grid)

    indices = random_points(10, len(inter))
    precision = 1e-1

    oper_y = inter.partial_y()
    cfs_y = np.dot(oper_y, cfs)
    values = inter.inverse()[indices] @ cfs_y
    expected = np.array([1.0] * len(indices))
    assert values == pytest.approx(expected, rel=precision)

    oper_xy = np.dot(oper_y, inter.partial_x())
    cfs_xy = np.dot(oper_xy, cfs)
    values = inter.inverse()[indices] @ cfs_xy
    expected = np.array([0.0] * len(indices))
    assert values == pytest.approx(expected, abs=precision)

    oper_yz = np.dot(oper_y, inter.partial_z())
    cfs_yz = np.dot(oper_yz, cfs)
    values = inter.inverse()[indices] @ cfs_yz
    expected = np.array([0.0] * len(indices))
    assert values == pytest.approx(expected, abs=precision)


def test_z_identity():
    """Test `Interpolation.partial_z` in a simple setting."""

    def z_identity(arg: Point | Grid) -> Real | ScalarValues:
        r = arg[0] if instance(arg, Point) else arg[:, 0]
        theta = arg[1] if instance(arg, Point) else arg[:, 1]
        # phi = arg[2] if instance(arg, Point) else arg[:, 2]
        return r * np.cos(theta)

    inter = Interpolation((10, 10))
    grid = inter.grid()
    cfs = inter.transform() @ z_identity(grid)

    indices = random_points(10, len(inter))
    precision = 1e-1

    oper_z = inter.partial_z()
    cfs_z = np.dot(oper_z, cfs)
    values = inter.inverse()[indices] @ cfs_z
    expected = np.array([1.0] * len(indices))
    assert values == pytest.approx(expected, rel=precision)

    oper_yz = np.dot(oper_z, inter.partial_y())
    cfs_yz = np.dot(oper_yz, cfs)
    values = inter.inverse()[indices] @ cfs_yz
    expected = np.array([0.0] * len(indices))
    assert values == pytest.approx(expected, abs=precision)

    oper_zx = np.dot(oper_z, inter.partial_x())
    cfs_zx = np.dot(oper_zx, cfs)
    values = inter.inverse()[indices] @ cfs_zx
    expected = np.array([0.0] * len(indices))
    assert values == pytest.approx(expected, abs=precision)


def test_partial_x(scalar, scalar_x):
    """Test `Interpolation.partial_x`"""
    inter = Interpolation((12, 144))
    grid = inter.grid()
    precision_abs = 2e-1

    indices = random_points(10, len(inter))
    points = grid[indices]

    oper_x = inter.partial_x()
    cfs = inter.transform() @ scalar(grid)
    cfs_x = np.dot(oper_x, cfs)
    values = inter.inverse()[indices] @ cfs_x
    expected = scalar_x(points)
    assert values == pytest.approx(expected, abs=precision_abs)


def test_partial_y(scalar, scalar_y):
    """Test `Interpolation.partial_y`"""
    inter = Interpolation((12, 144))
    grid = inter.grid()
    precision_abs = 2e-1

    indices = random_points(10, len(inter))
    points = grid[indices]

    oper_y = inter.partial_y()
    cfs = inter.transform() @ scalar(grid)
    cfs_y = np.dot(oper_y, cfs)
    values = inter.inverse()[indices] @ cfs_y
    expected = scalar_y(points)
    assert values == pytest.approx(expected, abs=precision_abs)


def test_partial_z(scalar, scalar_z):
    """Test `Interpolation.partial_z`"""
    inter = Interpolation((12, 144))
    precision_abs = 2e-1

    indices = random_points(10, len(inter))
    points = inter.grid()[indices]

    oper_z = inter.partial_z()
    cfs = inter.transform() @ scalar(inter.grid())
    cfs_z = np.dot(oper_z, cfs)
    values = inter.inverse()[indices] @ cfs_z
    expected = scalar_z(points)
    assert values == pytest.approx(expected, abs=precision_abs)
