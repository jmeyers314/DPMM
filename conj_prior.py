# Equation numbers refer to Kevin Murphy's "Conjugate Bayesian analysis of the Gaussian
# distribution" note unless otherwise specified.

from operator import mul
import numpy as np
from scipy.stats import norm, multivariate_normal, invwishart
from scipy.special import gamma


def gammad(d, nu_over_2):
    """D-dimensional gamma function."""
    nu = 2.0 * nu_over_2
    return np.pi**(d*(d-1.)/4)*np.multiply.reduce([gamma(0.5*(nu+1-i)) for i in range(d)])


def vmv(vec, mat=None, vec2=None):
    """Multiply a vector times a matrix times a vector, with a transpose applied to the first
    vector.  This is so common, I functionized it.

    @param vec  The first vector (will be transposed).
    @param mat  The matrix in the middle.  Identity by default.
    @param vec2 The second vector (will not be transposed.)  By default, the same as the vec.
    @returns    Product.  Could be a scalar or a matrix depending on whether vec is a row or column
                vector.
    """
    if mat is None:
        mat = np.eye(len(vec))
    if vec2 is None:
        vec2 = vec
    return np.dot(vec.T, np.dot(mat, vec2))


def multivariate_t(d, nu, mu, Sig, x=None):
    """Return multivariate t distribution: t_nu(x | mu, Sig), in d-dimensions.  If x is None,
    return a callable."""
    detSig = np.linalg.det(Sig)
    invSig = np.linalg.inv(Sig)
    coef = gamma(nu/2.0+d/2.0) * detSig**(-0.5)
    coef /= gamma(nu/2.0) * nu**(d/2.0)*np.pi**(d/2.0)

    def f(x):
        return coef * (1.0 + 1./nu*vmv((x-mu).T, invSig))**(-(nu+d)/2.0)
    if x is None:
        return f
    else:
        return f(x)


class ConjugatePrior(object):
    """
    theta = parameters of the model.
    x = data point.
    D = {x} = all data.
    params = parameters of the prior.
    """
    def sample(self, size=None):
        """Return one or more samples from prior distribution."""
        raise NotImplementedError

    def like1(self, *args, **kwargs):
        """Return likelihood for single data element.  Pr(x | theta)"""
        raise NotImplementedError

    def likelihood(self, *args, **kwargs):
        # It's quite likely overriding this will yield faster results...
        """Returns Pr(D | theta).  If D is None, then returns a callable."""

        D = kwargs.pop('D', None)

        def l(Ds):
            like1 = self.like1(*args, **kwargs)
            return reduce(mul, (like1(D1) for D1 in Ds), 1.0)

        if D is None:
            return l
        else:
            return l(D)

    def __call__(self, *args):
        """Returns Pr(theta), i.e. the prior probability."""
        raise NotImplementedError

    def post_params(self, D):
        """Returns new parameters for updating prior->posterior by initializing new object."""
        raise NotImplementedError

    def post(self, D):
        """Returns new ConjugatePrior with updated params for prior->posterior."""
        return type(self)(*self.post_params(D))

    def pred(self, x):
        """Prior predictive.  Pr(x | params)"""
        raise NotImplementedError

    def post_pred(self, D, x=None):
        """Posterior predictive.  Pr(x | D)"""
        if x is None:
            return self.post(D).pred
        else:
            return self.post(D).pred(x)


class NIW(ConjugatePrior):
    """Normal-Inverse-Wishart prior for multivariate Gaussian distribution.

    Model parameters
    ----------------
    mu :    multivariate mean
    Sigma : covariance matrix

    Prior parameters
    ----------------
    mu_0
    kappa_0
    Lam_0
    nu_0
    """
    def __init__(self, mu_0, kappa_0, Lam_0, nu_0):
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.Lam_0 = Lam_0
        self.nu_0 = nu_0
        self.d = len(mu_0)

    def _S(self, D):
        """Scatter matrix.  D is [NOBS, NDIM].  Returns [NDIM, NDIM] array."""
        # Eq (244)
        Dbar = np.mean(D, axis=0)
        return vmv((D-Dbar))

    def sample(self, size=1):
        """Return a sample {mu, Sigma} or list of samples [{mu_1, Sigma_1}, ...] from
        distribution.
        """
        Sig = invwishart.rvs(df=self.nu_0, scale=np.linalg.inv(self.Lam_0), size=size)
        if size == 1:
            return multivariate_normal.rvs(self.mu_0, Sig/self.kappa_0), Sig
        else:
            return zip((multivariate_normal.rvs(self.mu_0, S/self.kappa_0) for S in Sig), Sig)

    def like1(self, mu, Sigma, x=None):
        """Returns likelihood Pr(x | mu, Sigma), for a single data point.  Returns callable if D1
        is None.
        """
        rv = multivariate_normal(mean=mu, cov=Sigma)
        if x is None:
            return rv.pdf
        else:
            return rv.pdf(x)

    def __call__(self, mu, Sigma):
        """Returns Pr(mu, Sigma), i.e., the prior."""
        # Eq (249)
        Z = (2**(0.5*self.nu_0*self.d) * gammad(self.d, 0.5*self.nu_0) *
             (2*np.pi/self.kappa_0)**(self.d/2) / np.linalg.det(self.Lam_0)**(0.5*self.nu_0))
        detSig = np.linalg.det(Sigma)
        invSig = np.linalg.inv(Sigma)
        # Eq (248)
        return 1./Z * detSig**(-(0.5*(self.nu_0+self.d)+1)) * np.exp(
            -0.5*np.trace(np.dot(self.Lam_0, invSig) - self.kappa_0/2*vmv(mu-self.mu_0, invSig)))

    def post_params(self, D):
        """Recall D is [NOBS, NDIM]."""
        Dbar = np.mean(D, axis=0)
        n = len(D)
        # Eq (252)
        kappa_n = self.kappa_0 + n
        # Eq (253)
        nu_n = self.nu_0 + n
        # Eq (251) (note typo in original, mu+0 -> mu_0)
        mu_n = (self.kappa_0 * self.mu_0 + n * Dbar) / kappa_n
        # Eq (254)
        Lam_n = (self.Lam_0 +
                 self._S(D) +
                 self.kappa_0*n/(self.kappa_0+n)*vmv((Dbar-self.mu_0).T))
        return mu_n, kappa_n, Lam_n, nu_n

    def pred(self, x):
        """Prior predictive.  Pr(x)"""
        return multivariate_t(self.d, self.nu_0-self.d+1, self.mu_0,
                              self.Lam_0*(self.kappa_0+1)/(self.kappa_0 - self.d + 1), x)

    def marginal_likelihood(self, D):
        """Return Pr(D) = \int Pr(D, theta) Pr(theta)"""
        # Eq (266)
        n = len(D)
        mu_n, kappa_n, Lam_n, nu_n = self.post_params(D)
        detLam0 = np.linalg.det(self.Lam_0)
        detLamn = np.linalg.det(Lam_n)
        num = gammad(self.d, self.nu_n/2.0) * detLam0**(self.nu_0/2.0)
        den = np.pi**(n*self.d/2.0) * gammad(self.d, self.nu_0/2.0) * detLamn**(self.nu_n/2.0)
        return num/den * (self.kappa_0/self.kappa_n)**(self.d/2.0)


class GaussianMeanKnownVariance(ConjugatePrior):
    """Model univariate Gaussian with known variance and unknown mean.

    Model parameters
    ----------------
    mu :    multivariate mean

    Prior parameters
    ----------------
    mu_0 :  prior mean
    sig_0 : prior standard deviation

    Fixed parameters
    ----------------
    sig : Known variance.  Treat as a prior parameter to make __init__() with with post_params(),
          post(), post_pred(), etc., though note this never actually gets updated.
    """
    def __init__(self, mu_0, sig_0, sig):
        self.mu_0 = mu_0
        self.sig_0 = sig_0
        self.sig = sig
        self._norm1 = np.sqrt(2*np.pi*self.sig**2)
        self._norm2 = np.sqrt(2*np.pi*self.sig_0**2)

    def sample(self, size=None):
        """Return a sample `mu` or samples [mu1, mu2, ...] from distribution."""
        return np.random.normal(self.mu_0, self.sig_0, size=size)

    def like1(self, mu, x):
        """Returns likelihood Pr(x | mu), for a single data point.
        """
        return np.exp(-0.5*(x-mu)**2/self.sig**2) / self._norm1

    def __call__(self, mu):
        """Returns Pr(mu), i.e., the prior."""
        # Slow
        # return norm(loc=self.mu_0, scale=self.sig_0).pdf(mu)
        return np.exp(-0.5*(mu-self.mu_0)**2/self.sig_0**2) / self._norm2

    def post_params(self, D):
        """Recall D is [NOBS]."""
        try:
            n = len(D)
        except TypeError:
            n = 1
        Dbar = np.mean(D)
        sigsqr_n = 1./(n/self.sig**2 + 1./self.sig_0**2)
        sig_n = np.sqrt(sigsqr_n)
        mu_n = sig_n**2 * (self.mu_0/self.sig_0**2 + n*Dbar/self.sig**2)
        return mu_n, sig_n, self.sig

    def pred(self, x):
        """Prior predictive.  Pr(x)"""
        # Again, would like to do the following, but it's slow.
        # return norm(loc=self.mu_0, scale=np.sqrt(self.sig**2+self.sig_0**2)).pdf(x)
        sig = self.sig**2 + self.sig_0**2
        return np.exp(-0.5*(x-self.mu_0)**2/sig**2) / (np.sqrt(2*np.pi)*sig)

    def marginal_likelihood(self, D):
        """Fully marginalized likelihood Pr(D)"""
        n = len(D)
        Dbar = np.sum(D)
        num = self.sig
        den = (np.sqrt(2*np.pi)*self.sig)**n*np.sqrt(n*self.sig_0**2+self.sig**2)
        exponent = -np.sum(D**2)/2*self.sig**2 - self.mu_0/(2*self.sig_0**2)
        expnum = self.sig_0**2*n**2*Dbar**2/self.sig**2 + self.sig**2*self.mu_0**2/self.sig_0**2
        expnum += 2*n*Dbar*self.mu_0
        expden = 2*(n*self.sig_0**2+self.sig**2)
        return num/den*np.exp(exponent+expnum/expden)
