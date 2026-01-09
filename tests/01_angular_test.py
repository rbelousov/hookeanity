"""Test `hookeanity.angular` module"""

import pytest
import numpy as np
from hookeanity.angular import fourier, SphericalHarmonic


def test_fourier():
    """Test `fourier()`"""
    phis = np.array([1, 2, 3])
    assert fourier(0, phis) == pytest.approx([1, 1, 1])
    assert fourier(2, phis) == pytest.approx(np.sqrt(2) * np.cos(2 * phis))
    assert fourier(-3, phis) == pytest.approx(np.sqrt(2) * np.sin(3 * phis))
    assert fourier(3, phis, derivative=True) == pytest.approx(
        -3 * np.sqrt(2) * np.sin(3 * phis)
    )
    assert fourier(-2, phis, derivative=True) == pytest.approx(
        2 * np.sqrt(2) * np.cos(2 * phis)
    )


def test_spherical_harmonic():
    """Test `SphericalHarmonic`"""
    sph = SphericalHarmonic((4, 3))
    grid = np.array(
        [[np.pi / 6, np.pi / 3], [np.pi / 3, np.pi / 6], [np.pi / 33, np.pi / 7]]
    )
    values = sph(grid)
    assert len(values) == 3
    assert values.tolist() == pytest.approx([0.191622, 0.0, -0.000336779], abs=1e-6)

    assert SphericalHarmonic.from_spectral_order(7) == 15

    nodes, weights = SphericalHarmonic.quadrature(7)
    basis = [
        SphericalHarmonic((ell, m)) for ell in range(7) for m in range(-ell, ell + 1)
    ]
    for i, fi in enumerate(basis):
        for j, fj in enumerate(basis):
            value = np.sum(weights * fi(nodes) * fj(nodes))
            if i == j:
                assert value == pytest.approx(1, abs=1e-6), f"{{{(i, j)}}}"
            else:
                assert value == pytest.approx(0, abs=1e-6), f"{{{(i, j)}}}"


def analytic(
    l: int, m: int, theta: np.ndarray, phi: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic formulas for testing

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
    c21 = np.sqrt(15 / (4 * pi))
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
        val = -c11 * np.sin(theta) * np.cos(phi)
        dth = -c11 * np.cos(theta) * np.cos(phi)
        dph = +c11 * np.sin(theta) * np.sin(phi)
    elif (l, m) == (1, -1):
        val = c11 * np.sin(theta) * np.sin(phi)
        dth = c11 * np.cos(theta) * np.sin(phi)
        dph = c11 * np.sin(theta) * np.cos(phi)
    elif (l, m) == (2, 0):
        val = c20 * (3 * np.cos(theta) ** 2 - 1)
        dth = -6 * c20 * np.sin(theta) * np.cos(theta)
        dph = np.zeros_like(theta)
    elif (l, m) == (2, 1):
        val = -c21 * np.sin(theta) * np.cos(theta) * np.cos(phi)
        dth = -c21 * (np.cos(theta) ** 2 - np.sin(theta) ** 2) * np.cos(phi)
        dph = +c21 * np.sin(theta) * np.cos(theta) * np.sin(phi)
    elif (l, m) == (2, 2):
        val = c22 * np.sin(theta) ** 2 * np.cos(2 * phi)
        dth = c22 * (2 * np.sin(theta) * np.cos(theta)) * np.cos(2 * phi)
        dph = -2 * c22 * np.sin(theta) ** 2 * np.sin(2 * phi)
    else:
        raise ValueError("Add more cases if needed.")

    return val, dth, dph


def test_derivatives():
    """Test derivatives of the spherical harmonics"""
    nodes, _ = SphericalHarmonic.quadrature(2)

    for l, m in [(0, 0), (1, 0), (1, 1), (1, -1), (2, 0), (2, 1), (2, 2)]:
        val_a, dth_a, dph_a = analytic(l, m, nodes[:, 0], nodes[:, 1])

        sh = SphericalHarmonic((l, m))
        val = sh(nodes)
        dth = sh.partial_theta(nodes)
        dph = sh.partial_phi(nodes)

        # tune tolerances as you like
        assert val == pytest.approx(val_a), f"Value mismatched at order {(l, m)}"
        assert dth == pytest.approx(dth_a), f"Derivative mismatched at order {(l, m)}"
        assert dph == pytest.approx(dph_a), f"Derivative mismatched at order {(l, m)}"
