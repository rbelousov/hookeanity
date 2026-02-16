"""Test `hookeanity.radial` module"""

import pytest
import numpy as np
from scipy.special import eval_jacobi
from hookeanity.base import is_real
from hookeanity.radial import jacobi_radau_quadrature, RadialPolynomial


def test_jacobi_radau_quadrature():
    """Test `jacobi_radau_quadrature()`"""
    num = 5
    nodes, weights = jacobi_radau_quadrature(num)
    assert len(nodes) == num
    assert len(weights) == len(nodes)

    def poly(m, x):
        return eval_jacobi(m, 0, 2, x)

    for m in range(num):
        for n in range(num):
            value = np.sum(weights * poly(m, nodes) * poly(n, nodes))
            if m == n:
                assert value == pytest.approx(8 / (2 * n + 3)), f"{{{(m, n)}}}"
            else:
                assert value == pytest.approx(0), f"{{{(m, n)}}}"


def test_radial_polynomial():
    """Test `RadialPolynomial`"""
    num = 5
    nodes, weights = RadialPolynomial.quadrature(num - 1)
    assert RadialPolynomial(0)(nodes) == pytest.approx(np.full_like(nodes, np.sqrt(3)))

    # Integral of 1:  ∫_0^1 r^2 dr = 1/3
    assert np.sum(weights) == pytest.approx(1 / 3)

    # Integral of r:  ∫_0^1 r^3 dr = 1/4
    assert np.sum(weights * nodes) == pytest.approx(1 / 4)

    basis = [RadialPolynomial(order) for order in range(num)]

    for m in range(num):
        for n in range(num):
            value = np.sum(weights * basis[m](nodes) * basis[n](nodes))
            if m == n:
                assert value == pytest.approx(1), f"{{{(m, n)}}}"
            else:
                assert value == pytest.approx(0), f"{{{(m, n)}}}"

    assert is_real(basis[2](nodes[1]))


def test_derivative():
    """Test evaluation of the derivative of the radial polynomial"""
    num = 5
    nodes, weights = RadialPolynomial.quadrature(num - 1)
    basis = [RadialPolynomial(order) for order in range(num)]

    poly = np.polynomial.Polynomial([1, 2, 3, 4, 5])
    coefs = [np.sum(weights * b(nodes) * poly(nodes)) for b in basis]
    vals = np.asarray(
        [
            np.sum([c * b(node, derivative=True) for c, b in zip(coefs, basis)])
            for node in nodes
        ]
    )
    assert vals == pytest.approx(poly.deriv()(nodes))

    assert is_real(basis[2](nodes[1], derivative=True))
