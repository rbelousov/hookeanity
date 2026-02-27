"""Test `hookeanity.angular` module"""

import pytest
from typing import cast
import numpy as np
from scipy.special import sph_legendre_p, legendre
from hookeanity.base import is_real
from hookeanity.angular import fourier, SphericalHarmonic, geodesic_cos
from hookeanity.angular import MaternInterpolation


def test_fourier():
    """Test `fourier()`"""
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
    assert is_real(one)
    assert one == pytest.approx(np.sqrt(2) * np.cos(2 * phi))


def test_spherical_legendre_p():
    """Test functionality of theta dependence in spherical harmonics"""
    thetas = np.pi * np.array([0, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 5 / 6, 1])
    phi = np.pi / 6
    angles = np.array([[theta, phi] for theta in thetas])
    vals = {
        (0, 0): [
            0.282095,
            0.282095,
            0.282095,
            0.282095,
            0.282095,
            0.282095,
            0.282095,
            0.282095,
            0.282095,
        ],
        (1, -1): [
            0,
            0.172747,
            0.244301,
            0.299207,
            0.345494,
            0.299207,
            0.244301,
            0.172747,
            0,
        ],
        (1, 0): [
            0.488603,
            0.423142,
            0.345494,
            0.244301,
            0,
            -0.244301,
            -0.345494,
            -0.423142,
            -0.488603,
        ],
        (1, 1): [
            0,
            -0.172747,
            -0.244301,
            -0.299207,
            -0.345494,
            -0.299207,
            -0.244301,
            -0.172747,
            0,
        ],
        (2, -2): [
            0,
            0.0965686,
            0.193137,
            0.289706,
            0.386274,
            0.289706,
            0.193137,
            0.0965686,
            0,
        ],
        (2, -1): [
            0,
            0.334523,
            0.386274,
            0.334523,
            0,
            -0.334523,
            -0.386274,
            -0.334523,
            0,
        ],
        (2, 0): [
            0.630783,
            0.394239,
            0.157696,
            -0.0788479,
            -0.315392,
            -0.0788479,
            0.157696,
            0.394239,
            0.630783,
        ],
        (2, 1): [
            0,
            -0.334523,
            -0.386274,
            -0.334523,
            0,
            0.334523,
            0.386274,
            0.334523,
            0,
        ],
        (2, 2): [
            0,
            0.0965686,
            0.193137,
            0.289706,
            0.386274,
            0.289706,
            0.193137,
            0.0965686,
            0,
        ],
    }

    ders = {
        (0, 0): [0, 0, 0, 0, 0, 0, 0, 0, 0],
        (1, -1): [
            0.345494,
            0.299207,
            0.244301,
            0.172747,
            0,
            -0.172747,
            -0.244301,
            -0.299207,
            -0.345494,
        ],
        (1, 0): [
            0,
            -0.244301,
            -0.345494,
            -0.423142,
            -0.488603,
            -0.423142,
            -0.345494,
            -0.244301,
            0,
        ],
        (1, 1): [
            -0.345494,
            -0.299207,
            -0.244301,
            -0.172747,
            0,
            0.172747,
            0.244301,
            0.299207,
            0.345494,
        ],
        (2, -2): [
            0,
            0.334523,
            0.386274,
            0.334523,
            0,
            -0.334523,
            -0.386274,
            -0.334523,
            0,
        ],
        (2, -1): [
            0.772548,
            0.386274,
            0,
            -0.386274,
            -0.772548,
            -0.386274,
            0,
            0.386274,
            0.772548,
        ],
        (2, 0): [
            0,
            -0.819411,
            -0.946175,
            -0.819411,
            0,
            0.819411,
            0.946175,
            0.819411,
            0,
        ],
        (2, 1): [
            -0.772548,
            -0.386274,
            0,
            0.386274,
            0.772548,
            0.386274,
            0,
            -0.386274,
            -0.772548,
        ],
        (2, 2): [
            0,
            0.334523,
            0.386274,
            0.334523,
            0,
            -0.334523,
            -0.386274,
            -0.334523,
            0,
        ],
    }

    for l in range(2):
        for m in range(-l, l + 1):
            val, der = sph_legendre_p(l, abs(m), thetas, diff_n=1)

            assert val == pytest.approx(vals[l, abs(m)], abs=5e-6), (
                f"Value mismatched at order {(l, m)}"
            )
            assert der == pytest.approx(ders[l, abs(m)], abs=5e-6), (
                f"Value mismatched at order {(l, m)}"
            )

            sph = SphericalHarmonic((l, m))
            ff = fourier(m, phi)
            assert sph(angles) == pytest.approx(
                ff * np.array(vals[l, abs(m)]), abs=5e-6
            ), f"Value mismatched at order {(l, m)}"
            assert sph.partial_theta(angles) == pytest.approx(
                ff * np.array(ders[l, abs(m)]), abs=5e-6
            ), f"Value mismatched at order {(l, m)}"


def test_spherical_harmonic():
    """Test `SphericalHarmonic`"""
    sph = SphericalHarmonic((4, 3))
    grid = np.array(
        [[np.pi / 6, np.pi / 3], [np.pi / 3, np.pi / 6], [np.pi / 33, np.pi / 7]]
    )
    values = cast(np.ndarray, sph(grid))
    assert len(values) == 3
    assert values.tolist() == pytest.approx([-0.191622, 0.0, 0.000336779], abs=1e-6)

    omega = [np.pi / 6, np.pi / 3]
    value = sph(omega)
    assert is_real(value)
    assert value == pytest.approx(-0.191622, abs=1e-6)

    nodes, weights = SphericalHarmonic.quadrature(7)
    basis = [
        SphericalHarmonic((ell, m)) for ell in range(7) for m in range(-ell, ell + 1)
    ]
    for i, fi in enumerate(basis):
        for j, fj in enumerate(basis):
            value = np.sum(weights * fi(nodes) * fj(nodes))
            if i == j:
                assert value == pytest.approx(1, abs=5e-3), f"{{{(i, j)}}}"
            else:
                assert value == pytest.approx(0, abs=5e-3), f"{{{(i, j)}}}"


def analytic_derivative(
    l: int, m: int, theta: np.ndarray, phi: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic formulas for testing derivatives of the spherical harmonics

    Args:
        l: Polar order
        m: Azimuthal order
        theta: Grid of polar angles
        phi: Grid of azimuthal angles

    Returns:
        Value of the spherical harmonic and its partial derivatives with respect to the polar and azimuthal spherical angles

    """
    pi = np.pi
    c00 = 1 / (2 * np.sqrt(pi))
    c10 = np.sqrt(3 / (4 * pi))
    c11 = np.sqrt(3 / (4 * pi))
    c20 = np.sqrt(5 / (16 * pi))
    c21 = np.sqrt(15 / (16 * pi))
    c22 = np.sqrt(15 / (16 * pi))
    if (l, m) == (0, 0):
        val = c00 * np.ones_like(theta)
        dth = np.zeros_like(theta)
        dph = np.zeros_like(theta)
    elif (l, m) == (1, 0):
        val = c10 * np.cos(theta)
        dth = -c10 * np.sin(theta)
        dph = np.zeros_like(theta)
    elif (l, m) == (1, 1):
        val = c11 * np.sin(theta) * np.cos(phi)
        dth = c11 * np.cos(theta) * np.cos(phi)
        dph = -c11 * np.sin(theta) * np.sin(phi)
    elif (l, m) == (1, -1):
        val = c11 * np.sin(theta) * np.sin(phi)
        dth = c11 * np.cos(theta) * np.sin(phi)
        dph = c11 * np.sin(theta) * np.cos(phi)
    elif (l, m) == (2, 0):
        val = c20 * (3 * np.cos(theta) ** 2 - 1)
        dth = -6 * c20 * np.sin(theta) * np.cos(theta)
        dph = np.zeros_like(theta)
    elif (l, m) == (2, 1):
        val = c21 * np.sin(2 * theta) * np.cos(phi)
        dth = 2 * c21 * np.cos(2 * theta) * np.cos(phi)
        dph = -c21 * np.sin(2 * theta) * np.sin(phi)
    elif (l, m) == (2, 2):
        val = c22 * np.sin(theta) ** 2 * np.cos(2 * phi)
        dth = 2 * c22 * np.sin(theta) * np.cos(theta) * np.cos(2 * phi)
        dph = -2 * c22 * np.sin(theta) ** 2 * np.sin(2 * phi)
    else:
        raise ValueError("Add more cases if needed.")

    return val, dth, dph


def test_derivatives():
    """Test derivatives of the spherical harmonics"""
    nodes, _ = SphericalHarmonic.quadrature(2)
    for l, m in [(0, 0), (1, 0), (1, 1), (1, -1), (2, 0), (2, 1), (2, 2)]:
        val_a, dth_a, dph_a = analytic_derivative(l, m, nodes[:, 0], nodes[:, 1])

        sh = SphericalHarmonic((l, m))
        val = sh(nodes)
        dth = sh.partial_theta(nodes)
        dph = sh.partial_phi(nodes)

        # tune tolerances as you like
        assert val == pytest.approx(val_a), f"Value mismatched at order {(l, m)}"
        assert dth == pytest.approx(dth_a), f"Derivative mismatched at order {(l, m)}"
        assert dph == pytest.approx(dph_a), f"Derivative mismatched at order {(l, m)}"


def test_quadrature():
    """Test convention of Lebedev quadrature"""
    nodes, _ = SphericalHarmonic.quadrature(5)

    assert np.all(nodes[:, 0] >= 0)
    assert np.all(nodes[:, 0] <= np.pi)
    assert np.all(nodes[:, 1] >= 0)
    assert np.all(nodes[:, 1] <= 2 * np.pi)


def test_legendre():
    """Test `scipy.special.legendre` for correctness"""
    nodes, _ = SphericalHarmonic.quadrature(5)
    for l in range(5):
        for node in nodes:
            ref = np.full_like(nodes, node)
            value = np.sum(
                [
                    SphericalHarmonic((l, m))(ref) * SphericalHarmonic((l, m))(nodes)
                    for m in range(-l, l + 1)
                ],
                axis=0,
            )
            expected = (2 * l + 1) / (4 * np.pi) * legendre(l)(geodesic_cos(ref, nodes))
            assert value == pytest.approx(expected, abs=1e-6), (
                f"Legendre polynomial mismatched at order {l}"
            )


def test_matern_interpolation():
    """Test `MaternInterpolation`"""
    nodes, weights = SphericalHarmonic.quadrature(5)
    inter = MaternInterpolation(nodes, weights=weights)

    precision = 1e-1
    matrix = np.einsum("y,yp,yq", inter._bandfilter, inter._values, inter._values)
    assert inter._coefficients @ matrix == pytest.approx(
        np.eye(len(inter)), abs=precision
    )

    precision = 2e-1
    sz = len(inter)
    for p in range(sz):
        lp = cast(np.ndarray, inter[p](nodes))
        assert np.sum(lp) == pytest.approx(np.sqrt(weights[p]) ** -1, abs=precision)
        assert lp[p] == pytest.approx(np.sqrt(weights[p]) ** -1, abs=precision)
        for q in range(sz):
            lq = inter[q](nodes)
            expected = 1.0 if p == q else 0.0
            assert np.einsum("n,n,n", weights, lp, lq) == pytest.approx(
                expected, abs=precision
            )

    precision = 1e-3
    oper_theta = inter.partial_theta()
    oper_phi = inter.partial_phi()
    rng = np.random.default_rng(seed=0)
    npoints = 10
    chi = inter.chi()
    for l, m in [(0, 0), (1, 0), (1, 1), (1, -1), (2, 0), (2, 1), (2, 2)]:
        points = rng.choice(list(range(len(nodes))), size=npoints, replace=False)

        sh = SphericalHarmonic((l, m))
        val = np.einsum("q,pq,q->p", weights, chi, sh(nodes))
        expected = np.sqrt(weights) * sh(nodes)
        assert val[points] == pytest.approx(expected[points])

        dth = np.sqrt(weights) * sh.partial_theta(nodes)
        sh_theta = oper_theta @ val
        assert sh_theta[points] == pytest.approx(dth[points], abs=precision), (
            f"Incorrect derivative at order {(l, m)}"
        )

        dph = np.sqrt(weights) * sh.partial_phi(nodes)
        sh_phi = oper_phi @ val
        assert sh_phi[points] == pytest.approx(dph[points], abs=precision), (
            f"Incorrect derivative at order {(l, m)}"
        )

    # print('\ninter: ', inter.__len__(), sz)
    # raise
