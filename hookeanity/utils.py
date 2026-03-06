from .base import Real

def young_poisson_to_lame(E: Real, nu: Real) -> tuple[Real, Real]:
    """Convert Young's modulus (E) and Poisson ratio (nu) to Lamé parameters (mu, lambda)

    Returns
        (mu, lambda) : tuple of floats
    """
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, lam


def lame_to_young_poisson(mu: Real, lam: Real) -> tuple[Real, Real]:
    """Convert Lamé parameters (mu, lambda) to Young's modulus (E) and Poisson ratio (nu)

    Returns
        (E, nu) : tuple of floats
    """
    E = mu * (3 * lam + 2 * mu) / (lam + mu)
    nu = lam / (2 * (lam + mu))
    return E, nu