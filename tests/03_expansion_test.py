"""Test `hookeanity.expansion` model"""

import pytest
import numpy as np
import scipy as sp
from itertools import product
from hookeanity.expansion import Expansion, ScalarFunction
from hookeanity.radial import RadialPolynomial
from hookeanity.angular import SphericalHarmonic
from typing import cast
from numpy.typing import ArrayLike


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


def test_generators():
    """Test generator methods of `Expansion`"""
    expansion = Expansion((3, 3))

    counter = 0
    for ps, _ in expansion.points_and_weights():
        counter += 1
        assert ps.shape == (3,)
    assert counter == len(expansion._radii) * len(expansion._angles)

    for bf in expansion.basis_functions():
        assert np.isfinite(bf(0.5, 0.1, 0.7))


def test_expand_scalar(scalar):
    """Test `Expansion.expand_scalar"""
    expansion = Expansion((3, 8))
    bf = expansion.basis_function(RadialPolynomial(2), SphericalHarmonic((3, -1)))
    cfs = expansion.expand_scalar_function(bf)
    fnc = expansion.to_scalar_function(cfs)
    assert len(cfs) == expansion.size
    assert np.sum(cfs) == pytest.approx(1, abs=1e-2)

    rng = np.random.default_rng(seed=0)
    npoints = 10
    points = rng.choice(list(expansion.points()), size=npoints, replace=False)

    # Test basis function
    for n, point in enumerate(points):
        assert fnc(*point) == pytest.approx(bf(*point), abs=1e-2), (
            f"Error at the point {n}/{expansion.npoints}"
        )

    # Test scalar
    cfs = expansion.expand_scalar_function(scalar)
    fnc = expansion.to_scalar_function(cfs)
    for n, point in enumerate(points):
        assert fnc(*point) == pytest.approx(scalar(*point), rel=1e-2), (
            f"Error at the point {n}/{expansion.npoints}"
        )

    expected = cfs
    vals = np.array([scalar(*pnt) for pnt in expansion.points()])
    cfs = expansion.expand_scalar_values(vals)
    assert cfs == pytest.approx(expected)


def test_partial_theta():
    """Test derivatives of spherical harmonics with respect to theta"""
    # Analytical values of the scalar product between harmonics as their partial derivatives wrt theta
    vals = {
        (0, 0, 1, 0): -1.36035,
        (1, -1, 2, -1): -1.31715,
        (1, 0, 2, 0): -2.28138,
        (1, 1, 2, 1): -1.31715,
        (2, -1, 1, -1): 0.658576,
        (2, 0, 1, 0): 0.380229,
        (2, 1, 1, 1): 0.658576,
    }
    precision = 1e-3

    expansion = Expansion((0, 25), (1, 3))
    for l in range(2):
        for m in range(-l, l + 1):
            bf = expansion.basis_function(
                RadialPolynomial(0), SphericalHarmonic((l, m))
            )
            for p in range(2):
                for q in range(-p, p + 1):
                    index = (l, m, p, q)
                    sph_pq = SphericalHarmonic((p, q))

                    def bf_theta(r, theta, phi):
                        return (
                            RadialPolynomial(0)(r)
                            * sph_pq.partial_theta([theta, phi])[0]
                        )

                    coef = np.sum(
                        [
                            wgt * bf(*pnt) * bf_theta(*pnt)
                            for pnt, wgt in expansion.points_and_weights()
                        ]
                    )
                    if index in vals:
                        assert coef == pytest.approx(vals[index], abs=precision)
                        continue
                    assert coef == pytest.approx(0, abs=precision), (
                        f"Element {index} is off"
                    )

    def lm_to_index(l, m):
        return l * l + (m + l)

    # def index_to_lm(idx):
    #     l = int(np.floor(np.sqrt(idx)))
    #     m = idx - l*l - l
    #     return l, m

    oper_theta = expansion.partial_theta()
    for l in range(2):
        for m in range(-l, l + 1):
            i = lm_to_index(l, m)

            for ll in range(2):
                for mm in range(-ll, ll + 1):
                    j = lm_to_index(ll, mm)
                    key = (l, m, ll, mm)

                    if key in vals:
                        assert oper_theta[i, j] == pytest.approx(
                            vals[key], abs=precision
                        )
                    else:
                        assert oper_theta[i, j] == pytest.approx(0.0, abs=precision)


def test_r_derivative():
    """Test derivative of the basis functions"""
    precision = 1e-4
    expansion = Expansion((10, 10), (5, 3))
    rng = np.random.default_rng(seed=0)
    npoints = 10
    npoly = 4
    nsph = 4
    points = rng.choice(
        cast(ArrayLike, list(expansion.points())), size=npoints, replace=False
    )
    polys = rng.choice(
        cast(ArrayLike, expansion._radial_functions), size=npoly, replace=False
    )
    sphs = rng.choice(
        cast(ArrayLike, expansion._angular_functions), size=nsph, replace=False
    )

    # Testing partial_r
    oper_r = expansion.partial_r()
    for poly in polys:
        for sph in sphs:
            expected_r = np.array(
                [poly(pnt[[0]], True)[0] * sph(pnt[None, 1:])[0] for pnt in points]
            )

            cfs = expansion.expand_scalar_function(expansion.basis_function(poly, sph))
            cfs_r = np.dot(oper_r, cfs)
            fnc_r = expansion.to_scalar_function(cfs_r)
            for n, (point, val) in enumerate(zip(points, expected_r)):
                assert fnc_r(*point) == pytest.approx(val, rel=precision), (
                    f"Error at the point {n}/{npoints} ({point})"
                )


def test_phi_derivative():
    """Test derivative of the basis functions"""
    precision = 1e-4
    expansion = Expansion((10, 10), (5, 3))
    rng = np.random.default_rng(seed=0)
    npoints = 10
    npoly = 4
    nsph = 4
    points = rng.choice(list(expansion.points()), size=npoints, replace=False)
    polys = rng.choice(
        cast(ArrayLike, expansion._radial_functions), size=npoly, replace=False
    )
    sphs = rng.choice(
        cast(ArrayLike, expansion._angular_functions), size=nsph, replace=False
    )

    # Testing partial_phi
    oper_phi = expansion.partial_phi()
    for poly in polys:
        for sph in sphs:
            expected_phi = np.array(
                [poly(pnt[[0]]) * sph.partial_phi(pnt[None, 1:]) for pnt in points]
            )

            cfs = expansion.expand_scalar_function(expansion.basis_function(poly, sph))
            cfs_phi = np.dot(oper_phi, cfs)
            fnc_phi = expansion.to_scalar_function(cfs_phi)
            for n, (point, val) in enumerate(zip(points, expected_phi)):
                assert fnc_phi(*point) == pytest.approx(val, rel=precision), (
                    f"Error at the point {n}/{npoints} ({point})"
                )


def test_theta_derivative():
    """Test derivative of the basis functions"""
    precision_abs = 1
    precision_rel = 0.3
    expansion = Expansion((5, 24))
    rng = np.random.default_rng(seed=0)
    npoints = 10
    npoly = 4
    nsph = 4
    points = rng.choice(
        cast(ArrayLike, list(expansion.points())), size=npoints, replace=False
    )
    polys = rng.choice(
        cast(ArrayLike, expansion._radial_functions), size=npoly, replace=False
    )
    sphs = rng.choice(
        cast(ArrayLike, expansion._angular_functions), size=nsph, replace=False
    )

    # Testing partial_theta
    oper_theta = expansion.partial_theta()
    for poly in polys:
        for sph in sphs:

            def bf_theta(r, theta, phi):
                return poly([r])[0] * sph.partial_theta([theta, phi])[0]

            expected_theta = np.array([bf_theta(*pnt) for pnt in points])

            cfs = expansion.expand_scalar_function(expansion.basis_function(poly, sph))
            cfs_theta = np.dot(oper_theta, cfs)
            fnc_theta = expansion.to_scalar_function(cfs_theta)
            for n, (point, val) in enumerate(zip(points, expected_theta)):
                if np.absolute(val) < precision_abs:
                    assert fnc_theta(*point) == pytest.approx(val, abs=precision_abs), (
                        f"Error at the point {n}/{npoints} ({point})"
                    )
                else:
                    assert fnc_theta(*point) == pytest.approx(val, rel=precision_rel), (
                        f"Error at the point {n}/{npoints} ({point})"
                    )


def test_derivatives(scalar, scalar_r, scalar_theta, scalar_phi):
    """Test spectral derivatives"""
    expansion = Expansion((10, 60), basis=(5, 5))
    cfs = expansion.expand_scalar_function(scalar)
    precision = 0.25

    rng = np.random.default_rng(seed=0)
    npoints = 100
    points = rng.choice(list(expansion.points()), size=npoints, replace=False)

    oper = expansion.partial_r()
    cfs_r = np.dot(oper, cfs)
    f_r = expansion.to_scalar_function(cfs_r)
    for n, point in enumerate(points):
        assert f_r(*point) == pytest.approx(scalar_r(*point), abs=precision), (
            f"Error at the point {n}/{npoints}"
        )

    oper_theta = expansion.partial_theta()
    cfs_theta = np.dot(oper_theta, cfs)
    f_theta = expansion.to_scalar_function(cfs_theta)
    for n, point in enumerate(points):
        assert f_theta(*point) == pytest.approx(scalar_theta(*point), abs=precision), (
            f"Error at the point {n}/{npoints}"
        )

    oper_phi = expansion.partial_phi()
    cfs_phi = np.dot(oper_phi, cfs)
    f_phi = expansion.to_scalar_function(cfs_phi)
    for n, point in enumerate(points):
        assert f_phi(*point) == pytest.approx(scalar_phi(*point), abs=precision), (
            f"Error at the point {n}/{npoints}"
        )

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
        return np.ones(r.shape)

    return wrapper

@pytest.fixture
def simple_scalar_yz() -> ScalarFunction:
    """Fixture of a test scalar function diffirentiated with respect to the x component"""

    def wrapper(r, theta, phi):
        return np.zeros(r.shape)

    return wrapper

def test_jacobian_x():
    """Test elements of the first column of the jacobian matrix"""
    expansion = Expansion((10, 10), basis=(5, 5))
    jxs = expansion.sample_scalar_function(expansion._jacobian_x)
    assert jxs.shape == (expansion.npoints, 3, 1)
    assert np.all(np.isfinite(jxs))
    # assert jxs[:, 0, 0].shape == (expansion.npoints,)

def test_partial_x(simple_scalar, simple_scalar_x):
    expansion = Expansion((5, 20))
    cfs = expansion.expand_scalar_function(simple_scalar)
    precision = 0.25

    rng = np.random.default_rng(seed=0)
    npoints = 100
    points = rng.choice(list(expansion.points()), size=npoints, replace=False)

    oper_x = expansion.partial_x()
    cfs_x = np.dot(oper_x, cfs)
    f_x = expansion.to_scalar_function(cfs_x)
    for n, point in enumerate(points):
        assert f_x(*point) == pytest.approx(simple_scalar_x(*point), abs=precision), (
            f"Error at the point {n}/{npoints}"
        )

def test_values_partial_x(simple_scalar, simple_scalar_x):
    expansion = Expansion((5, 20))
    cfs = expansion.expand_scalar_function(simple_scalar)
    precision = 0.25

    rng = np.random.default_rng(seed=0)
    npoints = 100
    indices = rng.choice(list(range(expansion.npoints)), size=npoints, replace=False)
    points = list(expansion.points())

    oper_x = expansion.partial_x()
    cfs_x = np.dot(oper_x, cfs)
    all_values = expansion.to_scalar_values(cfs_x, indices)

    assert all_values.shape == (100,)

    for value, index in zip(all_values, indices):
        assert value == pytest.approx(simple_scalar_x(*points[index]), abs=precision), (
            f"Error at index point {index}"
        )

def test_values_partial_y(simple_scalar, simple_scalar_yz):
    expansion = Expansion((5, 20))
    cfs = expansion.expand_scalar_function(simple_scalar)
    precision = 0.25

    rng = np.random.default_rng(seed=0)
    npoints = 100
    indices = rng.choice(list(range(expansion.npoints)), size=npoints, replace=False)
    points = list(expansion.points())

    oper_y = expansion.partial_y()
    cfs_y = np.dot(oper_y, cfs)
    all_values = expansion.to_scalar_values(cfs_y, indices)

    assert all_values.shape == (100,)

    for value, index in zip(all_values, indices):
        assert value == pytest.approx(simple_scalar_yz(*points[index]), abs=precision), (
            f"Error at index point {index}"
        )

def test_values_partial_z(simple_scalar, simple_scalar_yz):
    expansion = Expansion((5, 20))
    cfs = expansion.expand_scalar_function(simple_scalar)
    precision = 0.25

    rng = np.random.default_rng(seed=0)
    npoints = 100
    indices = rng.choice(list(range(expansion.npoints)), size=npoints, replace=False)
    points = list(expansion.points())

    oper_z = expansion.partial_z()
    cfs_z = np.dot(oper_z, cfs)
    all_values = expansion.to_scalar_values(cfs_z, indices)

    assert all_values.shape == (100,)

    for value, index in zip(all_values, indices):
        assert value == pytest.approx(simple_scalar_yz(*points[index]), abs=precision), (
            f"Error at index point {index}"
        )

# def test_jacobian():
#     """Test elements of the jacobian"""
#     expansion = Expansion((10, 10), basis=(5, 5))

#     # print('\n\n', expansion._angles, '\n\n')

#     theta_xa = expansion._theta_x_angular()
#     assert np.all(np.isfinite(theta_xa))

#     phi_xa = expansion._phi_x_angular()
#     assert np.all(np.isfinite(phi_xa))


# def test_xyz_derivative(scalar, scalar_x):
#     """Test Cartesian derivatives"""
#     expansion = Expansion((10, 10))
#     cfs = expansion.expand_scalar_function(scalar)
#     precision = 0.25
#     assert np.all(expansion._radii > 0)

#     rng = np.random.default_rng(seed=0)
#     npoints = 100
#     points = rng.choice(list(expansion.points()), size=npoints, replace=False)

#     oper_x = expansion.partial_x()
#     cfs_x = np.dot(oper_x, cfs)

#     f_x = expansion.to_scalar_function(cfs_x)

#     for n, point in enumerate(points):
#         assert f_x(*point) == pytest.approx(scalar_x(*point), abs=precision), (
#             f"Error at the point {n}/{npoints}"
#         )
