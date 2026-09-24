"""Test `hookeanity.utils` module."""

import pytest
from hookeanity.utils import lame_to_young_poisson, young_poisson_to_lame


def test_young_poisson_to_lame():
    """Test conversion of Young modulus and Poisson ratio to Lamé coefficients."""
    modulus = 1000.0
    nu = 0.3

    lam, mu = young_poisson_to_lame(modulus, nu)

    assert mu == pytest.approx(modulus / (2 * (1 + nu)))
    assert lam == pytest.approx(modulus * nu / ((1 + nu) * (1 - 2 * nu)))


def test_lame_to_young_poisson():
    """Test conversion of Lamé coefficients to Young modulus and Poisson ratio."""
    mu = 384.61538461538464
    lam = 576.9230769230769

    modulus, nu = lame_to_young_poisson(lam, mu)

    assert modulus == pytest.approx(1000.0)
    assert nu == pytest.approx(0.3)


@pytest.mark.parametrize(
    "modulus, nu",
    [
        (1000.0, 0.3),
        (200.0, 0.25),
        (1e6, 0.1),
        (5000.0, 0.45),
    ],
)
def test_young_poisson(modulus, nu):
    """Test reversibility of conversion from Young modulus and Possion ratio to Lamé coefficients."""
    lam, mu = young_poisson_to_lame(modulus, nu)
    E_back, nu_back = lame_to_young_poisson(lam, mu)

    assert E_back == pytest.approx(modulus)
    assert nu_back == pytest.approx(nu)


@pytest.mark.parametrize(
    "mu, lam",
    [
        (1.0, 1.0),
        (10.0, 5.0),
        (384.61538461538464, 576.9230769230769),
        (1000.0, 2000.0),
    ],
)
def test_lame(mu, lam):
    """Test reversibility of conversion from Lamé coefficients to Young modulus and Possion ratio."""
    modulus, nu = lame_to_young_poisson(lam, mu)
    lam_back, mu_back = young_poisson_to_lame(modulus, nu)

    assert mu_back == pytest.approx(mu)
    assert lam_back == pytest.approx(lam)
