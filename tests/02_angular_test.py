"""Test `hookeanity.angular` module."""

import pytest
from typing import cast
import numpy as np
from scipy.special import sph_legendre_p, legendre
from hookeanity.base import Real, instance
from hookeanity.angular import (
    fibonacci_points,
    fibonacci_weight,
    gauss_legendre_fourier_points,
    gauss_legendre_fourier_weights,
    fourier,
    SphericalHarmonic,
    geodesic_cos,
)
from hookeanity.angular import MaternInterpolator


def test_fourier():
    """Test real fourier functions."""
    phis = np.array([1, 2, 3])
    assert fourier(0, phis) == pytest.approx([1, 1, 1])
    assert fourier(2, phis) == pytest.approx(np.sqrt(2) * np.cos(2 * phis))
    assert fourier(-3, phis) == pytest.approx(-np.sqrt(2) * np.sin(3 * phis))
    assert fourier(3, phis, derivative=True) == pytest.approx(
        3 * np.sqrt(2) * np.sin(3 * phis)
    )
    assert fourier(-2, phis, derivative=True) == pytest.approx(
        2 * np.sqrt(2) * np.cos(2 * phis)
    )

    phi = 2
    one = fourier(2, phi)
    assert instance(one, Real)
    assert one == pytest.approx(np.sqrt(2) * np.cos(2 * phi))


def test_spherical_legendre_p():
    """Test functionality of the theta dependence in spherical harmonics."""
    thetas = np.pi * np.array([0, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 5 / 6, 1])
    phi = np.pi / 6
    angles = np.array([[theta, phi] for theta in thetas])
    order = (2, -1)
    vals = np.array([0, -0.3345, -0.3863, -0.3345, 0, 0.3345, 0.3863, 0.3345, 0])
    ders = np.array([-0.7725, -0.3863, 0, 0.3863, 0.7725, 0.3863, 0, -0.3863, -0.7725])

    val, der = sph_legendre_p(order[0], abs(order[1]), thetas, diff_n=1)
    assert val == pytest.approx(vals, abs=5e-5)
    assert der == pytest.approx(ders, abs=5e-5)

    sph = SphericalHarmonic(order)
    ff = fourier(order[1], phi)
    assert sph(angles) == pytest.approx(ff * np.array(vals), abs=5e-5)
    assert sph.partial_theta(angles) == pytest.approx(ff * ders, abs=5e-5)


@pytest.fixture
def simple_grid():
    """Return a simple grid of solid angles for testing."""
    return np.array(
        [[np.pi / 6, np.pi / 3], [np.pi / 3, np.pi / 6], [np.pi / 33, np.pi / 7]]
    )


def test_spherical_harmonic(simple_grid):
    """Test real spherical harmonics."""
    sph = SphericalHarmonic((4, 3))
    values = cast(np.ndarray, sph(simple_grid))
    assert len(values) == 3
    assert values.tolist() == pytest.approx([-0.191622, 0.0, 0.000336779], abs=1e-6)

    omega = [np.pi / 6, np.pi / 3]
    value = sph(omega)
    assert instance(value, Real)
    assert value == pytest.approx(-0.191622, abs=1e-6)


def analytic_derivative(
    ell: int, m: int, theta: np.ndarray, phi: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic formulas for testing derivatives of the spherical harmonics.

    Args:
        ell: Polar order
        m: Azimuthal order
        theta: Grid of polar angles
        phi: Grid of azimuthal angles

    Returns:
        Value of the spherical harmonic and its partial derivatives with respect to the polar and azimuthal spherical angles

    Raises:
        ValueError: If the order (ell, m) is not implemented
    """
    pi = np.pi
    c00 = 1 / (2 * np.sqrt(pi))
    c10 = np.sqrt(3 / (4 * pi))
    c11 = np.sqrt(3 / (4 * pi))
    c20 = np.sqrt(5 / (16 * pi))
    c21 = np.sqrt(15 / (16 * pi))
    c22 = np.sqrt(15 / (16 * pi))
    if (ell, m) == (0, 0):
        val = c00 * np.ones_like(theta)
        dth = np.zeros_like(theta)
        dph = np.zeros_like(theta)
    elif (ell, m) == (1, 0):
        val = c10 * np.cos(theta)
        dth = -c10 * np.sin(theta)
        dph = np.zeros_like(theta)
    elif (ell, m) == (1, 1):
        val = c11 * np.sin(theta) * np.cos(phi)
        dth = c11 * np.cos(theta) * np.cos(phi)
        dph = -c11 * np.sin(theta) * np.sin(phi)
    elif (ell, m) == (1, -1):
        val = c11 * np.sin(theta) * np.sin(phi)
        dth = c11 * np.cos(theta) * np.sin(phi)
        dph = c11 * np.sin(theta) * np.cos(phi)
    elif (ell, m) == (2, 0):
        val = c20 * (3 * np.cos(theta) ** 2 - 1)
        dth = -6 * c20 * np.sin(theta) * np.cos(theta)
        dph = np.zeros_like(theta)
    elif (ell, m) == (2, 1):
        val = c21 * np.sin(2 * theta) * np.cos(phi)
        dth = 2 * c21 * np.cos(2 * theta) * np.cos(phi)
        dph = -c21 * np.sin(2 * theta) * np.sin(phi)
    elif (ell, m) == (2, 2):
        val = c22 * np.sin(theta) ** 2 * np.cos(2 * phi)
        dth = 2 * c22 * np.sin(theta) * np.cos(theta) * np.cos(2 * phi)
        dph = -2 * c22 * np.sin(theta) ** 2 * np.sin(2 * phi)
    else:
        raise ValueError("Add more cases if needed.")

    return val, dth, dph


def test_harmonics_derivatives(simple_grid):
    """Test derivatives of the spherical harmonics."""
    for ell, m in [(0, 0), (1, 0), (1, 1), (1, -1), (2, 0), (2, 1), (2, 2)]:
        val_a, dth_a, dph_a = analytic_derivative(
            ell, m, simple_grid[:, 0], simple_grid[:, 1]
        )

        sh = SphericalHarmonic((ell, m))
        val = sh(simple_grid)
        dth = sh.partial_theta(simple_grid)
        dph = sh.partial_phi(simple_grid)

        # tune tolerances as you like
        assert val == pytest.approx(val_a), f"Value mismatched at order {(ell, m)}"
        assert dth == pytest.approx(dth_a), f"Derivative mismatched at order {(ell, m)}"
        assert dph == pytest.approx(dph_a), f"Derivative mismatched at order {(ell, m)}"

    ell, m = (1, -1)
    node = simple_grid[-1]
    val_a, dth_a, dph_a = analytic_derivative(ell, m, node[0], node[1])
    sh = SphericalHarmonic((ell, m))
    val = sh(node)
    dth = sh.partial_theta(node)
    dph = sh.partial_phi(node)
    assert val == pytest.approx(val_a)
    assert dth == pytest.approx(dth_a)
    assert dph == pytest.approx(dph_a)
    assert instance(val, Real)
    assert instance(dth, Real)
    assert instance(dph, Real)


def test_fibonacci():
    """Test fibanacci quadrature."""
    n = 100
    nodes = fibonacci_points(n)
    weight = fibonacci_weight(n)

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


def test_gauss_legendre_fourier_grid():
    """Test Gauss-Legendre/Fourier quadrature."""
    n_lat, n_lon = 8, 16
    nodes = gauss_legendre_fourier_points(n_lat)
    weights = gauss_legendre_fourier_weights(n_lat)

    basis = [
        SphericalHarmonic((ell, m)) for ell in range(6) for m in range(-ell, ell + 1)
    ]
    for i, fi in enumerate(basis):
        for j, fj in enumerate(basis):
            value = np.einsum("q,q,q", weights, fi(nodes), fj(nodes))
            if i == j:
                assert value == pytest.approx(1, abs=1e-10), f"{{{(i, j)}}}"
            else:
                assert value == pytest.approx(0, abs=1e-10), f"{{{(i, j)}}}"

    assert len(nodes) == n_lat * n_lon
    assert np.sum(weights) == pytest.approx(4 * np.pi)


def test_legendre(simple_grid):
    """Test a summation formula for spherical harmonics."""
    for ell in range(5):
        for ref in simple_grid:
            value = np.sum(
                [
                    SphericalHarmonic((ell, m))(ref)
                    * SphericalHarmonic((ell, m))(simple_grid)
                    for m in range(-ell, ell + 1)
                ],
                axis=0,
            )
            expected = (
                (2 * ell + 1)
                / (4 * np.pi)
                * legendre(ell)(geodesic_cos(ref, simple_grid))
            )
            assert value == pytest.approx(expected, abs=1e-6), (
                f"Legendre polynomial mismatched at order {ell}"
            )


def test_matern_interpolation():
    """Test `MaternInterpolator`."""
    angles = fibonacci_points(100)
    weights = np.full(len(angles), fibonacci_weight(len(angles)))
    inter = MaternInterpolator(angles, weights=weights)

    precision = 2e-1
    sz = len(inter)
    for p in range(sz):
        lp = cast(np.ndarray, inter[p](angles))
        assert np.sum(lp) == pytest.approx(np.sqrt(weights[p]) ** -1, abs=precision)
        assert lp[p] == pytest.approx(np.sqrt(weights[p]) ** -1, abs=precision)
        for q in range(sz):
            lq = inter[q](angles)
            expected = 1.0 if p == q else 0.0
            assert np.einsum("n,n,n", weights, lp, lq) == pytest.approx(
                expected, abs=precision
            )

    precision = 1e-3
    oper_theta = inter.partial_theta()
    oper_phi = inter.partial_phi()
    rng = np.random.default_rng(seed=0)
    npoints = 10
    points = rng.choice(list(range(len(angles))), size=npoints, replace=False)
    chi = inter.chi()
    for ell, m in [(0, 0), (1, 0), (1, 1), (1, -1), (2, 0), (2, 1), (2, 2)]:
        sh = SphericalHarmonic((ell, m))
        val = np.einsum("q,pq,q->p", weights, chi, sh(angles))
        expected = np.sqrt(weights) * sh(angles)
        assert val[points] == pytest.approx(expected[points])

        dth = np.sqrt(weights) * sh.partial_theta(angles)
        sh_theta = oper_theta @ val
        assert sh_theta[points] == pytest.approx(dth[points], abs=precision), (
            f"Incorrect derivative at order {(ell, m)}"
        )

        dph = np.sqrt(weights) * sh.partial_phi(angles)
        sh_phi = oper_phi @ val
        assert sh_phi[points] == pytest.approx(dph[points], abs=precision), (
            f"Incorrect derivative at order {(ell, m)}"
        )
