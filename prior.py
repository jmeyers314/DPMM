# Equation numbers refer to Kevin Murphy's "Conjugate Bayesian analysis of the Gaussian
# distribution" note unless otherwise specified.

from operator import mul
import numpy as np
from scipy.special import gamma


def vTmv(vec, mat=None, vec2=None):
    """Multiply a vector transpose times a matrix times a vector.

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


def gammad(d, nu_over_2):
    """D-dimensional gamma function."""
    nu = 2.0 * nu_over_2
    return np.pi**(d*(d-1.)/4)*np.multiply.reduce([gamma(0.5*(nu+1-i)) for i in range(d)])


def multivariate_t(d, nu, mu, Sig, x=None):
    """Return multivariate t distribution: t_nu(x | mu, Sig), in d-dimensions.  If x is None,
    return a callable."""
    detSig = np.linalg.det(Sig)
    invSig = np.linalg.inv(Sig)
    coef = gamma(nu/2.0+d/2.0) * detSig**(-0.5)
    coef /= gamma(nu/2.0) * nu**(d/2.0)*np.pi**(d/2.0)

    def f(x):
        return coef * (1.0 + 1./nu*vTmv((x-mu).T, invSig))**(-(nu+d)/2.0)
    if x is None:
        return f
    else:
        return f(x)


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


def random_wish(dof, S, size=1):
    dim = S.shape[0]
    if size == 1:
        x = np.random.multivariate_normal(np.zeros(dim), S, size=dof)
        return np.dot(x.T, x)
    else:
        out = np.empty((size, dim, dim), dtype=np.float64)
        for i in range(size):
            x = np.random.multivariate_normal(np.zeros(dim), S, size=dof)
            out[i] = np.dot(x.T, x)
        return out


def random_invwish(dof, invS, size=1):
    return np.linalg.inv(random_wish(dof, invS, size=size))


class Prior(object):
    """
    theta = parameters of the model.
    x = data point.
    D = {x} = all data.
    params = parameters of the prior.
    """
    def __init__(self, post=None, *args, **kwargs):
        # Assume conjugate prior by default, i.e. that posterior is same form as prior
        if post is None:
            post = type(self)
        self._post = post

    def sample(self, size=None):
        """Return one or more samples from prior distribution."""
        raise NotImplementedError

    def like1(self, *args, **kwargs):
        """Return likelihood for single data element.  Pr(x | theta)"""
        raise NotImplementedError

    def likelihood(self, *args, **kwargs):
        # It's quite likely overriding this will yield faster results...
        """Returns Pr(D | theta)."""

        try:
            D = kwargs.pop('D')
        except KeyError:
            raise ValueError("Likelihood called without data.")

        return reduce(mul, (self.like1(*args, x=x, **kwargs) for x in D), 1.0)

    def __call__(self, *args):
        """Returns Pr(theta), i.e. the prior probability."""
        raise NotImplementedError

    def post_params(self, D):
        """Returns new parameters for updating prior->posterior by initializing new object."""
        raise NotImplementedError

    def post(self, D):
        """Returns new Prior with updated params for prior->posterior."""
        return self._post(*self.post_params(D))

    def pred(self, x):
        """Prior predictive.  Pr(x | params)"""
        raise NotImplementedError

    def post_pred(self, D, x=None):
        """Posterior predictive.  Pr(x | D)"""
        if x is None:
            return self.post(D).pred
        else:
            return self.post(D).pred(x)


class NIW(Prior):
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
        super(NIW, self).__init__()

    def _S(self, D):
        """Scatter matrix.  D is [NOBS, NDIM].  Returns [NDIM, NDIM] array."""
        # Eq (244)
        Dbar = np.mean(D, axis=0)
        return vTmv((D-Dbar))

    def sample(self, size=1):
        """Return a sample {mu, Sigma} or list of samples [{mu_1, Sigma_1}, ...] from
        distribution.
        """
        Sig = random_invwish(dof=self.nu_0, invS=self.Lam_0, size=size)
        if size == 1:
            return np.random.multivariate_normal(self.mu_0, Sig/self.kappa_0), Sig
        else:
            return zip((np.random.multivariate_normal(self.mu_0, S/self.kappa_0) for S in Sig),
                       Sig)

    def like1(self, mu, Sigma, x):
        """Returns likelihood Pr(x | mu, Sigma), for a single data point."""
        norm = (2*np.pi*np.linalg.det(Sigma))**(0.5*self.d)
        return np.exp(-0.5*vTmv(x-mu, np.linalg.inv(Sigma))) / norm

    def __call__(self, mu, Sigma):
        """Returns Pr(mu, Sigma), i.e., the prior."""
        # Eq (249)
        Z = (2**(0.5*self.nu_0*self.d) * gammad(self.d, 0.5*self.nu_0) *
             (2*np.pi/self.kappa_0)**(self.d/2) / np.linalg.det(self.Lam_0)**(0.5*self.nu_0))
        detSig = np.linalg.det(Sigma)
        invSig = np.linalg.inv(Sigma)
        # Eq (248)
        return 1./Z * detSig**(-(0.5*(self.nu_0+self.d)+1)) * np.exp(
            -0.5*np.trace(np.dot(self.Lam_0, invSig) - self.kappa_0/2*vTmv(mu-self.mu_0, invSig)))

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
                 self.kappa_0*n/(self.kappa_0+n)*vTmv((Dbar-self.mu_0).T))
        return mu_n, kappa_n, Lam_n, nu_n

    def pred(self, x):
        """Prior predictive.  Pr(x)"""
        return multivariate_t(self.d, self.nu_0-self.d+1, self.mu_0,
                              self.Lam_0*(self.kappa_0+1)/(self.kappa_0 - self.d + 1), x)

    def evidence(self, D):
        """Return Pr(D) = \int Pr(D | theta) Pr(theta)"""
        # Eq (266)
        n = len(D)
        mu_n, kappa_n, Lam_n, nu_n = self.post_params(D)
        detLam0 = np.linalg.det(self.Lam_0)
        detLamn = np.linalg.det(Lam_n)
        num = gammad(self.d, self.nu_n/2.0) * detLam0**(self.nu_0/2.0)
        den = np.pi**(n*self.d/2.0) * gammad(self.d, self.nu_0/2.0) * detLamn**(self.nu_n/2.0)
        return num/den * (self.kappa_0/self.kappa_n)**(self.d/2.0)


class GaussianMeanKnownVariance(Prior):
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
        super(GaussianMeanKnownVariance, self).__init__()

    def sample(self, size=None):
        """Return a sample `mu` or samples [mu1, mu2, ...] from distribution."""
        if size is None:
            return (np.random.normal(self.mu_0, self.sig_0),)
        else:
            return np.random.normal(self.mu_0, self.sig_0, size=size)

    def like1(self, mu, x):
        """Returns likelihood Pr(x | mu), for a single data point.
        """
        return np.exp(-0.5*(x-mu)**2/self.sig**2) / self._norm1

    def __call__(self, mu):
        """Returns Pr(mu), i.e., the prior."""
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
        sig = self.sig**2 + self.sig_0**2
        return np.exp(-0.5*(x-self.mu_0)**2/sig**2) / (np.sqrt(2*np.pi)*sig)

    def evidence(self, D):
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


class NIX(Prior):
    """Normal-Inverse-Chi-Square model for univariate Gaussian with params for mean and variance.

    Model parameters
    ----------------
    mu :    mean
    var :   variance

    Prior parameters
    ----------------
    mu_0 :  prior mean
    kappa_0 : belief in mu_0
    sigsqr_0 : prior variance
    nu_0 : belief in sigsqr_0
    """
    def __init__(self, mu_0, kappa_0, sigsqr_0, nu_0):
        self.mu_0 = float(mu_0)
        self.kappa_0 = float(kappa_0)
        self.sigsqr_0 = float(sigsqr_0)
        self.nu_0 = float(nu_0)
        super(NIX, self).__init__()

    def sample(self, size=1):
        var = 1./np.random.chisquare(df=self.nu_0, size=size)*self.sigsqr_0  # sanity check this.
        if size == 1:
            return np.random.normal(self.mu_0, var/self.kappa_0), var
        else:
            return zip((np.random.normal(self.mu_0, v/self.kappa_0) for v in var), var)

    def like1(self, mu, var, x):
        """Returns likelihood Pr(x | mu, var), for a single data point."""
        return np.exp(-0.5*(x-mu)**2/var) / np.sqrt(2*np.pi*var)

    def __call__(self, mu, var):
        """Returns Pr(mu, var), i.e., the prior density."""
        return (normal_density(self.mu_0, var/self.kappa_0, mu) *
                scaled_IX_density(self.nu_0, self.sigsqr_0, var))

    def post_params(self, D):
        try:
            n = len(D)
        except TypeError:
            n = 1
        Dbar = np.mean(D)
        kappa_n = self.kappa_0 + n
        mu_n = (self.kappa_0*self.mu_0 + n*Dbar)/kappa_n
        nu_n = self.nu_0 + n
        sigsqr_n = ((self.nu_0*self.sigsqr_0 + np.sum((D-Dbar)**2) +
                    n*self.kappa_0/(self.kappa_0+n)*(self.mu_0-Dbar)**2)/nu_n)
        return mu_n, kappa_n, sigsqr_n, nu_n

    def pred(self, x):
        """Prior predictive.  Pr(x)"""
        return t_density(self.nu_0, self.mu_0, (1.+self.kappa_0)*self.sigsqr_0/self.kappa_0, x)

    def evidence(self, D):
        """Fully marginalized likelihood Pr(D)"""
        mu_n, kappa_n, sigsqr_n, nu_n = self.post_params(D)
        try:
            n = len(D)
        except:
            n = 1
        return (gamma(nu_n/2.0)/gamma(self.nu_0/2.0) * np.sqrt(self.kappa_0/kappa_n) *
                (self.nu_0*self.sigsqr_0)**(self.nu_0/2.0) /
                (nu_n*sigsqr_n)**(nu_n/2.0) /
                np.pi**(n/2.0))


class NIG(Prior):
    """Normal-Inverse-Gamma prior for univariate Gaussian with params for mean and variance.

    Model parameters
    ----------------
    mu :    mean
    var :   variance

    Prior parameters
    ----------------
    mu_0 :  prior mean
    V_0
    a_0, b_0 : gamma parameters
    """
    def __init__(self, m_0, V_0, a_0, b_0):
        self.m_0 = float(m_0)
        self.V_0 = float(V_0)
        self.a_0 = float(a_0)
        self.b_0 = float(b_0)
        super(NIG, self).__init__()

    def sample(self, size=1):
        var = 1./np.random.gamma(self.a_0, self.b_0, size=size)
        if size == 1:
            return np.random.normal(self.m_0, self.V_0*np.sqrt(var)), var
        else:
            return zip(np.random.normal(self.m_0, self.V_0*np.sqrt(var), size=size), var)

    def like1(self, mu, var, x):
        """Returns likelihood Pr(x | mu, var), for a single data point."""
        return np.exp(-0.5*(x-mu)**2/var) / np.sqrt(2*np.pi*var)

    def __call__(self, mu, var):
        """Returns Pr(mu, var), i.e., the prior density."""
        normal = np.exp(-0.5*(self.m_0-mu)**2/(var*self.V_0))/np.sqrt(2*np.pi*var*self.V_0)
        ig = self.b_0**self.a_0/gamma(self.a_0)*var**(-(self.a_0+1))*np.exp(-self.b_0/var)
        return normal*ig

    def post_params(self, D):
        try:
            n = len(D)
        except TypeError:
            n = 1
        Dbar = np.mean(D)
        invV_0 = 1./self.V_0
        V_n = 1./(invV_0 + n)
        m_n = V_n*(invV_0*self.m_0 + n*Dbar)
        a_n = self.a_0 + n/2.0
        # The commented line below is from Murphy.  It doesn't pass the unit tests so I derived my
        # own formula which does.
        # b_n = self.b_0 + 0.5*(self.m_0**2*invV_0 + np.sum(Dbar**2) - m_n**2/V_n)
        b_n = self.b_0 + 0.5*(np.sum((D-Dbar)**2)+n/(1.0+n*self.V_0)*(self.m_0-Dbar)**2)
        return m_n, V_n, a_n, b_n

    def pred(self, x):
        """Prior predictive.  Pr(x)"""
        return t_density(2.0*self.a_0, self.m_0, self.b_0*(1.0+self.V_0)/self.a_0, x)

    def evidence(self, D):
        """Fully marginalized likelihood Pr(D)"""
        m_n, V_n, a_n, b_n = self.post_params(D)
        try:
            n = len(D)
        except:
            n = 1
        return (np.sqrt(np.abs(V_n/self.V_0)) * (self.b_0**self.a_0)/(b_n**a_n) *
                gamma(a_n)/gamma(self.a_0) / (np.pi**(n/2.0)*2.0**(n/2.0)))
