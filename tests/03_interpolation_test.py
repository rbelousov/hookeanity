"""Tests for `hookeanity.interpolation` module"""

import pytest
import numpy as np
import scipy as sp
from hookeanity.angular import SphericalHarmonic
from hookeanity.interpolation import ScalarFunction, Interpolation, fibonacci_points


def test_fibonacci():
    """Test fibanacci quadrature"""
    nodes = fibonacci_points(100)
    weight = 4 * np.pi / nodes.shape[0]

    basis = [
        SphericalHarmonic((ell, m)) for ell in range(7) for m in range(-ell, ell + 1)
    ]
    for i, fi in enumerate(basis):
        for j, fj in enumerate(basis):
            value = weight * np.sum(fi(nodes) * fj(nodes))
            if i == j:
                assert value == pytest.approx(1, abs=1e-1), f"{{{(i, j)}}}"
            else:
                assert value == pytest.approx(0, abs=1e-1), f"{{{(i, j)}}}"

    assert len(nodes) == 100


@pytest.fixture
def scalar() -> ScalarFunction:
    """Fixture of a test scalar function"""

    def wrapper(r, theta, phi):
        return sp.special.j0(r * np.cos(theta) * np.sin(phi))

    return wrapper


@pytest.fixture
def scalar_r() -> ScalarFunction:
    """Fixture of a test scalar function diffirentiated with respect to the radial component"""

    def wrapper(r, theta, phi):
        angular_factor = np.cos(theta) * np.sin(phi)
        return -sp.special.j1(r * angular_factor) * angular_factor

    return wrapper


@pytest.fixture
def scalar_theta() -> ScalarFunction:
    """Fixture of a test scalar function diffirentiated with respect to the azimuthal component"""

    def wrapper(r, theta, phi):
        return (
            sp.special.j1(r * np.cos(theta) * np.sin(phi)) * np.sin(theta) * np.sin(phi)
        )

    return wrapper


@pytest.fixture
def scalar_phi() -> ScalarFunction:
    """Fixture of a test scalar function diffirentiated with respect to the azimuthal component"""

    def wrapper(r, theta, phi):
        return (
            -sp.special.j1(r * np.cos(theta) * np.sin(phi))
            * np.cos(theta)
            * np.cos(phi)
        )

    return wrapper


def test_transform():
    """Test `Interpolation.transform()` and `Interpolation.inverse()`"""
    inter = Interpolation((5, 5))
    transform = inter.transform()
    inverse = inter.inverse()
    assert transform.shape == (inter.npoints, inter.npoints)
    assert inverse.shape == (inter.npoints, inter.npoints)
    assert inverse @ transform == pytest.approx(np.eye(transform.shape[-1]))


def test_partial_r(scalar, scalar_r):
    """Test `Interpolation.partial_r`"""
    inter = Interpolation((5, 5))
    values = inter.sample_function(scalar)
    modes = inter.transform() @ values
    modes_r = inter.partial_r() @ modes
    values_r = inter.inverse() @ modes_r
    expected = inter.sample_function(scalar_r)

    precision = 1e-3
    assert values_r == pytest.approx(expected, abs=precision)


def test_partial_theta(scalar, scalar_theta):
    """Test `Interpolation.partial_theta`"""
    inter = Interpolation((5, 5))
    values = inter.sample_function(scalar)
    modes = inter.transform() @ values
    modes_theta = inter.partial_theta() @ modes
    values_theta = inter.inverse() @ modes_theta
    expected = inter.sample_function(scalar_theta)

    rng = np.random.default_rng(seed=0)
    indices = rng.choice(range(inter.npoints), size=10, replace=False)
    for indx in indices:
        val = expected[indx]
        if abs(val) < 1:
            assert values_theta[indx] == pytest.approx(val, abs=5e-1)
        else:
            assert values_theta[indx] == pytest.approx(val, rel=5e-2)


def test_partial_phi(scalar, scalar_phi):
    """Test `Interpolation.partial_phi`"""
    inter = Interpolation((5, 5))
    values = inter.sample_function(scalar)
    modes = inter.transform() @ values
    modes_phi = inter.partial_phi() @ modes
    values_phi = inter.inverse() @ modes_phi
    expected = inter.sample_function(scalar_phi)

    rng = np.random.default_rng(seed=0)
    indices = rng.choice(range(inter.npoints), size=10, replace=False)
    for indx in indices:
        val = expected[indx]
        if abs(val) < 1:
            assert values_phi[indx] == pytest.approx(val, abs=5e-1)
        else:
            assert values_phi[indx] == pytest.approx(val, rel=5e-2)


@pytest.fixture
def simple_scalar() -> ScalarFunction:
    """Fixture of a test scalar function"""

    def wrapper(r, theta, phi):
        return r * np.sin(theta) * np.cos(phi)

    return wrapper


@pytest.fixture
def simple_scalar_x() -> ScalarFunction:
    """Fixture of a test scalar function diffirentiated with respect to the x component"""

    def wrapper(r, theta, phi):
        return 1.0  # np.ones(r.shape)

    return wrapper


@pytest.fixture
def simple_scalar_yz() -> ScalarFunction:
    """Fixture of a test scalar function diffirentiated with respect to the x component"""

    def wrapper(r, theta, phi):
        return 0.0  # np.zeros(r.shape)

    return wrapper


def test_jacobian_x():
    """Test elements of the first column of the jacobian matrix"""
    inter = Interpolation((10, 10))
    jxs = inter.sample_function(inter._jacobian_x)
    assert jxs.shape == (inter.npoints, 3)
    assert np.all(np.isfinite(jxs))
    assert jxs[:, 0].shape == (inter.npoints,)


# def test_partial_x(simple_scalar, simple_scalar_x):
#     inter = Interpolation((5, 5))
#     cfs = inter.expand_scalar_function(simple_scalar)
#     precision = 0.25

#     rng = np.random.default_rng(seed=0)
#     npoints = 100
#     points = rng.choice(list(inter.points()), size=npoints, replace=False)

#     oper_x = inter.partial_x()
#     cfs_x = np.dot(oper_x, cfs)
#     f_x = inter.to_scalar_function(cfs_x) # to define
#     for n, point in enumerate(points):
#         assert f_x(*point) == pytest.approx(simple_scalar_x(*point), abs=precision), (
#             f"Error at the point {n}/{npoints}"
#         )


def test_expand_scalar_function(scalar):
    """Test `Interpolation.expand_scalar_function` and `Interpolation.expand_to_scalar_values`"""
    inter = Interpolation((5, 5))
    cfs = inter.expand_scalar_function(scalar)
    values = inter.sample_function(scalar)
    cfs_expected = inter.transform() @ values
    assert cfs == pytest.approx(cfs_expected)

    indices = list(range(inter.npoints))
    points = list(inter.points())
    approx = inter.to_scalar_values(cfs, indices)
    for value, index in zip(approx, indices):
        assert value == pytest.approx(scalar(*points[index]))


def test_values_partial_x(simple_scalar, simple_scalar_x):
    """Test `Interpolation.partial_x`"""
    inter = Interpolation((5, 5))
    precision = 1e-1

    T = inter.transform()
    scalar_values = inter.sample_function(simple_scalar)
    cfs_transform = T @ scalar_values
    cfs = inter.expand_scalar_function(simple_scalar)
    assert cfs == pytest.approx(cfs_transform)

    rng = np.random.default_rng(seed=0)
    npoints = 10
    indices = rng.choice(list(range(inter.npoints)), size=npoints, replace=False)
    points = list(inter.points())

    oper_x = inter.partial_x()
    cfs_x = np.dot(oper_x, cfs)
    values = inter.inverse() @ cfs_x
    for index in indices:
        assert values[index] == pytest.approx(
            simple_scalar_x(*points[index]), rel=precision
        ), f"Error at {points[index]}"

    all_values = inter.to_scalar_values(cfs_x, indices)

    assert all_values.shape == (npoints,)

    for value, index in zip(all_values, indices):
        assert value == pytest.approx(simple_scalar_x(*points[index]), rel=precision), (
            f"Error at index point {index}"
        )


def test_values_partial_y(simple_scalar, simple_scalar_yz):
    """Test `Interpolation.partial_y`"""
    inter = Interpolation((5, 5))
    cfs = inter.expand_scalar_function(simple_scalar)
    precision = 1e-1

    rng = np.random.default_rng(seed=0)
    npoints = 10
    indices = rng.choice(list(range(inter.npoints)), size=npoints, replace=False)
    points = list(inter.points())

    oper_y = inter.partial_y()
    cfs_y = np.dot(oper_y, cfs)
    all_values = inter.to_scalar_values(cfs_y, indices)

    assert all_values.shape == (npoints,)

    for value, index in zip(all_values, indices):
        assert value == pytest.approx(
            simple_scalar_yz(*points[index]), abs=precision
        ), f"Error at index point {points[index]}"


def test_values_partial_z(simple_scalar, simple_scalar_yz):
    """Test `Interpolation.partial_z`"""
    inter = Interpolation((5, 5))
    cfs = inter.expand_scalar_function(simple_scalar)
    precision = 1e-1

    rng = np.random.default_rng(seed=0)
    npoints = 10
    indices = rng.choice(list(range(inter.npoints)), size=npoints, replace=False)
    points = list(inter.points())

    oper_z = inter.partial_z()
    cfs_z = np.dot(oper_z, cfs)
    all_values = inter.to_scalar_values(cfs_z, indices)

    assert all_values.shape == (npoints,)

    for value, index in zip(all_values, indices):
        assert value == pytest.approx(
            simple_scalar_yz(*points[index]), abs=precision
        ), f"Error at index point {points[index]}"


@pytest.fixture
def scalar_x(scalar_r, scalar_theta, scalar_phi) -> ScalarFunction:
    """Fixture of a test scalar function diffirentiated with respect to the x component"""

    def wrapper(r, theta, phi):
        r_x = np.sin(theta) * np.cos(phi)
        theta_x = np.cos(theta) * np.cos(phi) / r
        phi_x = -np.sin(phi) / (r * np.sin(theta))
        return (
            r_x * scalar_r(r, theta, phi)
            + theta_x * scalar_theta(r, theta, phi)
            + phi_x * scalar_phi(r, theta, phi)
        )

    return wrapper


@pytest.fixture
def scalar_y(scalar_r, scalar_theta, scalar_phi) -> ScalarFunction:
    """Fixture of a test scalar function differentiated with respect to the y component"""

    def wrapper(r, theta, phi):
        r_y = np.sin(theta) * np.sin(phi)
        theta_y = np.cos(theta) * np.sin(phi) / r
        phi_y = np.cos(phi) / (r * np.sin(theta))
        return (
            r_y * scalar_r(r, theta, phi)
            + theta_y * scalar_theta(r, theta, phi)
            + phi_y * scalar_phi(r, theta, phi)
        )

    return wrapper


@pytest.fixture
def scalar_z(scalar_r, scalar_theta, scalar_phi) -> ScalarFunction:
    """Fixture of a test scalar function differentiated with respect to the z component"""

    def wrapper(r, theta, phi):
        r_z = np.cos(theta)
        theta_z = -np.sin(theta) / r
        phi_z = 0.0
        return (
            r_z * scalar_r(r, theta, phi)
            + theta_z * scalar_theta(r, theta, phi)
            + phi_z * scalar_phi(r, theta, phi)
        )

    return wrapper


def test_bessel_partial_x(scalar, scalar_x):
    """Test `Interpolation.partial_x`"""
    inter = Interpolation((10, 10))
    precision_abs = 5e-1
    precision_rel = 1e-1

    cfs = inter.expand_scalar_function(scalar)

    rng = np.random.default_rng(seed=0)
    npoints = 10
    indices = rng.choice(list(range(inter.npoints)), size=npoints, replace=False)
    points = list(inter.points())

    oper_x = inter.partial_x()
    cfs_x = np.dot(oper_x, cfs)
    values = inter.inverse() @ cfs_x
    for index in indices:
        if abs(scalar_x(*points[index])) < 1.0:
            assert values[index] == pytest.approx(
                scalar_x(*points[index]), abs=precision_abs
            ), f"Error at {points[index]}"
        else:
            assert values[index] == pytest.approx(
                scalar_x(*points[index]), rel=precision_rel
            ), f"Error at {points[index]}"


def test_bessel_partial_y(scalar, scalar_y):
    """Test `Interpolation.partial_y`"""
    inter = Interpolation((10, 10))
    precision_abs = 5e-1
    precision_rel = 1e-1

    cfs = inter.expand_scalar_function(scalar)

    rng = np.random.default_rng(seed=0)
    npoints = 10
    indices = rng.choice(list(range(inter.npoints)), size=npoints, replace=False)
    points = list(inter.points())

    oper_x = inter.partial_x()
    cfs_x = np.dot(oper_x, cfs)
    values = inter.inverse() @ cfs_x
    for index in indices:
        if abs(scalar_y(*points[index])) < 1.0:
            assert values[index] == pytest.approx(
                scalar_y(*points[index]), abs=precision_abs
            ), f"Error at {points[index]}"
        else:
            assert values[index] == pytest.approx(
                scalar_y(*points[index]), rel=precision_rel
            ), f"Error at {points[index]}"


def test_bessel_partial_z(scalar, scalar_z):
    """Test `Interpolation.partial_z`"""
    inter = Interpolation((10, 10))
    precision_abs = 5e-1
    precision_rel = 1e-1

    cfs = inter.expand_scalar_function(scalar)

    rng = np.random.default_rng(seed=0)
    npoints = 10
    indices = rng.choice(list(range(inter.npoints)), size=npoints, replace=False)
    points = list(inter.points())

    oper_x = inter.partial_x()
    cfs_x = np.dot(oper_x, cfs)
    values = inter.inverse() @ cfs_x
    for index in indices:
        if abs(scalar_z(*points[index])) < 1.0:
            assert values[index] == pytest.approx(
                scalar_z(*points[index]), abs=precision_abs
            ), f"Error at {points[index]}"
        else:
            assert values[index] == pytest.approx(
                scalar_z(*points[index]), rel=precision_rel
            ), f"Error at {points[index]}"
