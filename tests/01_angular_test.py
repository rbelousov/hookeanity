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

    nodes, weights = SphericalHarmonic.quadrature(15)
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
