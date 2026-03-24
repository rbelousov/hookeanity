"""Utility modules for parameter conversion."""

from .base import Real


def young_poisson_to_lame(modulus: Real, ratio: Real) -> tuple[Real, Real]:
    """Convert Young modulus and Poisson ratio to Lamé parameters.

    Args:
        modulus: Young modulus
        ratio: Poisson ratio
    Returns:
        Lamé coefficients lambda and mu
    """
    lam = modulus * ratio / ((1 + ratio) * (1 - 2 * ratio))
    mu = modulus / (2 * (1 + ratio))
    return lam, mu


def lame_to_young_poisson(
    lam: Real,
    mu: Real,
) -> tuple[Real, Real]:
    """Convert Lamé parameters to Young modulus and Poisson ratio.

    Args:
        mu: shear modulus
        lam: first Lamé coefficient
    Returns
        Young modulus and Poisson ratio
    """
    modulus = mu * (3 * lam + 2 * mu) / (lam + mu)
    nu = lam / (2 * (lam + mu))
    return modulus, nu
