"""Test `hookeanity.utils` module."""

import pytest
from hookeanity.utils import young_poisson_to_lame, lame_to_young_poisson


def test_young_poisson_to_lame_known_values():
    """Test Young and Poisson to Lamé conversion."""
    modulus = 1000.0
    nu = 0.3

    lam, mu = young_poisson_to_lame(modulus, nu)

    assert mu == pytest.approx(modulus / (2 * (1 + nu)))
    assert lam == pytest.approx(modulus * nu / ((1 + nu) * (1 - 2 * nu)))


def test_lame_to_young_poisson_known_values():
    """Test Lamé to Young and Poisson conversion."""
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
def test_round_trip_young_poisson_to_lame_and_back(modulus, nu):
    """Test invertibility from Young and Possion."""
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
def test_round_trip_lame_to_young_poisson_and_back(mu, lam):
    """Test invertibility from Lamé coefficients."""
    modulus, nu = lame_to_young_poisson(lam, mu)
    lam_back, mu_back = young_poisson_to_lame(modulus, nu)

    assert mu_back == pytest.approx(mu)
    assert lam_back == pytest.approx(lam)
