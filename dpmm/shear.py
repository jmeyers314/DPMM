import numpy as np


def unshear(D, g):
    D1, D2 = D[..., 0], D[..., 1]  # real and imag parts of the data.
    g1, g2 = g[0], g[1]  # real and imag parts of the shear.
    a, b = D1 - g1, D2 - g2
    c, d = (1.0 - g1*D1 - g2*D2), g2*D1 - g1*D2
    # Now divide (a + bi) / (c + di) = 1/(c*c + d*d) * ((ac+bd) + i(bc - ad))
    out = np.empty_like(D)
    den = c**2 + d**2
    out[..., 0] = (a*c + b*d)/den
    out[..., 1] = (b*c - a*d)/den
    return out


def draw_g_1d_weak_shear(D, phi, label):
    """Update the estimate of the shear g.
    D here is the *unmanipulated* data.
    Assume that phi represents variance of a Gaussian distribution (can we make this more
    generic?)

    In the weak shear limit, we can Gibbs update g, since \Prod Pr(g | e_int_i, sigma_e_i) is a
    product of Gaussians.  Even if we're not in the weak shear limit, though, this is a reasonable
    way to generate proposals.
    """
    Lam = 0.0  # Use the canonical representation of a Gaussian.
    eta = 0.0
    for i, ph in enumerate(phi):
        index = np.nonzero(label == i)
        Lam += len(index[0])/ph
        eta += np.sum(D[index]/ph)
    var = 1./Lam
    mu = eta*var
    return np.random.normal(loc=mu, scale=np.sqrt(var))


def draw_g_2d_weak_shear(D, phi, label):
    """Update the estimate of the shear g.
    D here is the *unmanipulated* data.
    Assume that phi represents variance of a Gaussian distribution (can we make this more
    generic?)  Note that this is a bit weird, since the shear is 2D.  What we're really saying is
    the covariance matrix is var*np.eye(2).  (I think this is "tied" in the scikit-learn lingo.)

    In the weak shear limit, we can Gibbs update g, since \Prod Pr(g | e_int_i, sigma_e_i) is a
    product of Gaussians.  Even if we're not in the weak shear limit, though, this is a reasonable
    way to generate proposals.
    """
    Lam = 0.0  # Use the canonical representation of a Gaussian.
    eta = 0.0
    for i, ph in enumerate(phi):
        index = np.nonzero(label == i)
        Lam += len(index[0])/ph
        eta += np.sum(D[index]/ph, axis=0)
    var = 1./Lam
    mu = eta*var
    return np.random.multivariate_normal(mean=mu, cov=var*np.eye(2))


class Linear1DShear(object):
    """Data manipulator for 1-dimensional shear in the weak shear limit.
    I.e., manipulates data given shear via e_obs = e_int + g.
    """
    def __init__(self, g):
        self.g = g

    def init(self, D):
        """A quick and dirty estimate of g is just the average over D."""
        self.g = np.mean(D)

    def __call__(self, D):
        """Return the manipulated data, i.e., the current estimate for the unsheared ellipticity."""
        return D - self.g

    def unmanip(self, D):
        """Reverse transformation from __call__."""
        return D + self.g

    def update(self, D, phi, label, prior):
        """Update the estimate of the shear g.
        D here is the *unmanipulated* data.
        Assume that phi represents variance of a Gaussian distribution (can we make this more
        generic?)

        In this case we can Gibbs update g, since \Prod Pr(g | e_int_i, sigma_e_i) is a product of
        Gaussians.
        """
        self.g = draw_g_1d_weak_shear(D, phi, label)


class WeakShear(object):
    """Data manipulator for 2-dimensional shear in the weak shear limit.
    I.e., manipulates data given shear via e_obs = e_int + g.
    """
    def __init__(self, g):
        # g should be a length-2 array representing the real and imag parts of the complex
        # representation of the reduced shear.
        self.g = g

    def init(self, D):
        """A quick and dirty estimate of g is just the average over D."""
        self.g = np.mean(D, axis=0)

    def __call__(self, D):
        """Return the manipulated data, i.e., the current estimate for the unsheared ellipticity."""
        return D - self.g

    def unmanip(self, D):
        """Reverse transformation from __call__."""
        return D + self.g

    def update(self, D, phi, label, prior):
        """Update the estimate of the shear g.
        D here is the *unmanipulated* data.
        Assume that phi represents variance of a Gaussian distribution (can we make this more
        generic?)
        """
        self.g = draw_g_2d_weak_shear(D, phi, label)


class Shear(object):
    """Data manipulator for 2-dimensional shear, *not* in the weak shear limit.
    I.e., manipulates data given shear via e_obs = (e_int + g) / (1 + g* e_int).
    """
    def __init__(self, g):
        # g should be a length-2 array representing the real and imag parts of the complex
        # representation of the reduced shear.
        self.g = g

    def init(self, D):
        """A quick and dirty estimate of g is just the average over D."""
        self.g = np.mean(D, axis=0)

    def __call__(self, D):
        """Return the manipulated data, i.e., the current estimate for the unsheared ellipticity."""
        return unshear(D, self.g)

    def unmanip(self, D):
        """Reverse transformation from __call__."""
        return unshear(D, -self.g)

    def update(self, D, phi, label, prior):
        """Update the estimate of the shear g.
        D here is the *unmanipulated* data.
        Assume that phi represents variance of a Gaussian distribution (can we make this more
        generic?)
        """
        # Pr(g | D, phi, label) is complicated, so we need to MH update it.
        # For a proposal, though, we can still use the weak shear limit.
        # Whoops!  The weak shear limit proposal doesn't lead to *any* acceptances when ngal is
        # large.  Need to try something more clever.
        # prop_g = draw_g_2d_weak_shear(D, phi, label)
        prop_g = np.random.multivariate_normal(mean=self.g, cov=np.eye(2)*0.0005**2)

        current_e_int = unshear(D, self.g)
        prop_e_int = unshear(D, prop_g)
        current_lnlike = 0.0
        prop_lnlike = 0.0
        for i, ph in enumerate(phi):
            index = label == i
            current_lnlike += prior.lnlikelihood(current_e_int[index], ph)
            prop_lnlike += prior.lnlikelihood(prop_e_int[index], ph)
        if prop_lnlike > current_lnlike:
            self.g = prop_g
        else:
            u = np.random.uniform()
            if u < np.exp(prop_lnlike - current_lnlike):
                self.g = prop_g
