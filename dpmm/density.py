import numpy as np
from scipy.special import gamma
from utils import vTmv


def multivariate_t_density(nu, mu, Sig, x):
    """Return multivariate t distribution: t_nu(x | mu, Sig), in d-dimensions."""
    detSig = np.linalg.det(Sig)
    invSig = np.linalg.inv(Sig)
    d = len(mu)
    coef = gamma(nu/2.0+d/2.0) * detSig**(-0.5)
    coef /= gamma(nu/2.0) * nu**(d/2.0)*np.pi**(d/2.0)
    x = np.array(x)
    if len(x.shape) == 1:
        return coef * (1.0 + 1./nu*vTmv((x-mu).T, invSig)[0, 0])**(-(nu+d)/2.0)
    else:
        prod = np.array([vTmv(x_.T, invSig)[0, 0] for x_ in (x-mu)])
        return coef * (1.0 + prod/nu)**(-(nu+d)/2.0)


def t_density(nu, mu, sigsqr, x):
    c = gamma((nu+1.)/2.)/gamma(nu/2.)/np.sqrt(nu*np.pi*sigsqr)
    return c*(1.0+1./nu*((x-mu)**2/sigsqr))**(-(1.+nu)/2.0)


def scaled_IX_density(nu, sigsqr, x):
    return (1.0/gamma(nu/2.0) *
            (nu*sigsqr/2.0)**(nu/2.0) *
            x**(-nu/2.0-1.0) *
            np.exp(-nu*sigsqr/(2.0*x)))


def normal_density(mu, var, x):
    return np.exp(-0.5*(x-mu)**2/var)/np.sqrt(2*np.pi*var)
