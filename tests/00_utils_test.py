import pytest

from hookeanity.utils import young_poisson_to_lame, lame_to_young_poisson


def test_young_poisson_to_lame_known_values():
    E = 1000.0
    nu = 0.3

    mu, lam = young_poisson_to_lame(E, nu)

    assert mu == pytest.approx(E / (2 * (1 + nu)))
    assert lam == pytest.approx(E * nu / ((1 + nu) * (1 - 2 * nu)))


def test_lame_to_young_poisson_known_values():
    mu = 384.61538461538464
    lam = 576.9230769230769

    E, nu = lame_to_young_poisson(mu, lam)

    assert E == pytest.approx(1000.0)
    assert nu == pytest.approx(0.3)


@pytest.mark.parametrize(
    "E, nu",
    [
        (1000.0, 0.3),
        (200.0, 0.25),
        (1e6, 0.1),
        (5000.0, 0.45),
    ],
)
def test_round_trip_young_poisson_to_lame_and_back(E, nu):
    mu, lam = young_poisson_to_lame(E, nu)
    E_back, nu_back = lame_to_young_poisson(mu, lam)

    assert E_back == pytest.approx(E)
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
    E, nu = lame_to_young_poisson(mu, lam)
    mu_back, lam_back = young_poisson_to_lame(E, nu)

    assert mu_back == pytest.approx(mu)
    assert lam_back == pytest.approx(lam)