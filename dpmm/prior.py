# Equation numbers refer to Kevin Murphy's "Conjugate Bayesian analysis of the Gaussian
# distribution" note unless otherwise specified.

from operator import mul
import numpy as np
from scipy.special import gamma
from utils import gammad, random_invwish
from density import multivariate_t_density, t_density, normal_density, scaled_IX_density


class Prior(object):
    """
    In general, `Prior` object models represent the proabilistic graphical model:
      psi -> theta -> [x]
    where psi are hyperparameters of a prior,
    theta are parameters of the model being constrained,
    and [x] are data.

    For example, for an InvGamma model, the data (x) are scalar samples, the model is to infer the
    variance (theta) of a generative Gaussian distribution with known mean, and alpha, beta (psi)
    are hyperparameters for the prior on the Gaussian variance.

    We define the following probability distributions for each `Prior` object:

    Likelihood:  Pr([x] | theta).  Note [x] is conditionally independent of psi given theta.
    Prior: Pr(theta | psi).
    Posterior: Pr(theta | x, psi)  By Bayes theorem, equal to:
               Pr(x | theta) Pr(theta | psi) / Pr (x | psi)
    Predictive (or evidence): Pr(x | psi) = \int Pr(x | theta) Pr(theta | psi) d(theta).
    """
    def __init__(self, post=None, *args, **kwargs):
        # Assume conjugate prior by default, i.e. that posterior is same form as prior
        if post is None:
            post = type(self)
        self._post = post

    def sample(self, size=None):
        """Return one or more samples of the model parameters from prior distribution."""
        raise NotImplementedError

    def like1(self, x, *args, **kwargs):
        """Return likelihood for single data element.  Pr(x | theta).  This is conditionally
        independent of the hyperparameters psi.  If more than one data element is passed, then the
        likelihood will be returned for each element."""
        raise NotImplementedError

    def likelihood(self, D, *args, **kwargs):
        # It's quite likely overriding this will yield faster results...
        """Returns Pr(D | theta).  Does not broadcast over theta!"""
        return np.prod(self.like1(D, *args, **kwargs))

    def lnlikelihood(self, D, *args, **kwargs):
        """Returns ln(Pr(D | theta)).  Does not broadcast over theta!"""
        return np.log(self.likelihood(D, *args, **kwargs))

    def __call__(self, *args):
        """Returns Pr(theta | psi), i.e. the prior probability."""
        raise NotImplementedError

    def _post_params(self, D):
        """Returns new hyperparameters psi' for updating prior->posterior.  Can be sent to
        constructor to initialize a new object."""
        raise NotImplementedError

    def post(self, D):
        """Returns new Prior object using updated hyperparameters psi, which is the posterior given
        the data D."""
        return self._post(*self._post_params(D))

    def pred(self, x):
        """Prior predictive.  Pr(x | params).  Integrates out theta."""
        raise NotImplementedError


class GaussianMeanKnownVariance(Prior):
    """Model univariate Gaussian with known variance and unknown mean.

    Model parameters
    ----------------
    mu :    multivariate mean

    Prior parameters
    ----------------
    mu_0 :  prior mean
    sigsqr_0 : prior variance

    Fixed parameters
    ----------------
    sigsqr : Known variance.  Treat as a prior parameter to make __init__() with with
             _post_params(), post(), post_pred(), etc., though note this never actually gets
             updated.
    """
    def __init__(self, mu_0, sigsqr_0, sigsqr):
        self.mu_0 = mu_0
        self.sigsqr_0 = sigsqr_0
        self.sigsqr = sigsqr
        self._norm1 = np.sqrt(2*np.pi*self.sigsqr)
        self._norm2 = np.sqrt(2*np.pi*self.sigsqr_0)
        super(GaussianMeanKnownVariance, self).__init__()

    def sample(self, size=None):
        """Return a sample `mu` or samples [mu1, mu2, ...] from distribution."""
        if size is None:
            return np.random.normal(self.mu_0, np.sqrt(self.sigsqr_0))
        else:
            return np.random.normal(self.mu_0, np.sqrt(self.sigsqr_0), size=size)

    def like1(self, x, mu):
        """Returns likelihood Pr(x | mu), for a single data point.
        """
        return np.exp(-0.5*(x-mu)**2/self.sigsqr) / self._norm1

    def __call__(self, mu):
        """Returns Pr(mu), i.e., the prior."""
        return np.exp(-0.5*(mu-self.mu_0)**2/self.sigsqr_0) / self._norm2

    def _post_params(self, D):
        """Recall D is [NOBS]."""
        try:
            n = len(D)
        except TypeError:
            n = 1
        Dbar = np.mean(D)
        sigsqr_n = 1./(n/self.sigsqr + 1./self.sigsqr_0)
        mu_n = sigsqr_n * (self.mu_0/self.sigsqr_0 + n*Dbar/self.sigsqr)
        return mu_n, sigsqr_n, self.sigsqr

    def pred(self, x):
        """Prior predictive.  Pr(x)"""
        sigsqr = self.sigsqr + self.sigsqr_0
        return np.exp(-0.5*(x-self.mu_0)**2/sigsqr) / np.sqrt(2*np.pi*sigsqr)

    # FIXME!
    # def evidence(self, D):
    #     """Fully marginalized likelihood Pr(D)"""
    #     try:
    #         n = len(D)
    #     except:
    #         n = 1
    #     # import ipdb; ipdb.set_trace()
    #     D = np.array(D)
    #     Dbar = np.sum(D)
    #     num = np.sqrt(self.sigsqr)
    #     den = (2*np.pi*self.sigsqr)**(n/2.0)*np.sqrt(n*self.sigsqr_0+self.sigsqr)
    #     exponent = -np.sum(D**2)/(2.0*self.sigsqr) - self.mu_0/(2.0*self.sigsqr_0)
    #     expnum = self.sigsqr_0*n**2*Dbar**2/self.sigsqr + self.sigsqr*self.mu_0**2/self.sigsqr_0
    #     expnum += 2.0*n*Dbar*self.mu_0
    #     expden = 2.0*(n*self.sigsqr_0+self.sigsqr)
    #     return num/den*np.exp(exponent+expnum/expden)


class InvGamma(Prior):
    """Inverse Gamma distribution.  Note this parameterization matches Murphy's, not wikipedia's."""
    def __init__(self, alpha, beta, mu):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        super(InvGamma, self).__init__()

    def sample(self, size=None):
        return 1./np.random.gamma(self.alpha, scale=self.beta, size=size)

    def like1(self, x, var):
        """Returns likelihood Pr(x | var), for a single data point."""
        return np.exp(-0.5*(x-self.mu)**2/var) / np.sqrt(2*np.pi*var)

    def __call__(self, var):
        """Returns Pr(var), i.e., the prior density."""
        al, be = self.alpha, self.beta
        return be**(-al)/gamma(al) * var**(-1.-al) * np.exp(-1./(be*var))

    def _post_params(self, D):
        try:
            n = len(D)
        except TypeError:
            n = 1
        al_n = self.alpha + n/2.0
        be_n = 1./(1./self.beta + 0.5*np.sum((np.array(D)-self.mu)**2))
        return al_n, be_n, self.mu

    def pred(self, x):
        """Prior predictive.  Pr(x)"""
        # Careful.  Use 1/beta/alpha to match Murphy, not wikipedia!
        return t_density(2*self.alpha, self.mu, 1./self.beta/self.alpha, x)

    def evidence(self, D):
        """Fully marginalized likelihood Pr(D)"""
        raise NotImplementedError


class InvGamma2D(Prior):
    """Inverse Gamma distribution, but for modeling 2D covariance matrices proportional to the
    identity matrix."""
    def __init__(self, alpha, beta, mu):
        self.alpha = alpha
        self.beta = beta
        self.mu = np.array(mu)
        assert len(mu) == 2
        super(InvGamma2D, self).__init__()

    def sample(self, size=None):
        return 1./np.random.gamma(self.alpha, scale=self.beta, size=size)

    def like1(self, x, var):
        """Returns likelihood Pr(x | var), for a single data point."""
        assert isinstance(x, np.ndarray)
        assert x.shape[-1] == 2
        return np.exp(-0.5*np.sum((x-self.mu)**2, axis=-1)/var) / (2*np.pi*var)

    def lnlikelihood(self, D, var):
        """Returns the log likelihood for data D"""
        return -0.5*np.sum((D-self.mu)**2)/var - D.shape[0]*np.log(2*np.pi*var)

    def __call__(self, var):
        """Returns Pr(var), i.e., the prior density."""
        al, be = self.alpha, self.beta
        return be**(-al)/gamma(al) * var**(-1.-al) * np.exp(-1./(be*var))

    def _post_params(self, D):
        try:
            n = len(D)
        except TypeError:
            n = 1
        al_n = self.alpha + n  # it's + n/2.0 in InvGamma, but in 2D it's + n.
        be_n = 1./(1./self.beta + 0.5*np.sum((np.array(D)-self.mu)**2))  # Same formula for beta.
        return al_n, be_n, self.mu

    def pred(self, x):
        """Prior predictive.  Pr(x)"""
        assert isinstance(x, np.ndarray)
        assert x.shape[-1] == 2
        # Generalized from InvGamma.  Tested numerically.
        return multivariate_t_density(2*self.alpha, self.mu, 1./self.beta/self.alpha*np.eye(2), x)

    def evidence(self, D):
        """Fully marginalized likelihood Pr(D)"""
        raise NotImplementedError


class NormInvChi2(Prior):
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
        super(NormInvChi2, self).__init__()

    model_dtype = np.dtype([('mu', float), ('var', float)])

    def sample(self, size=None):
        if size is None:
            var = 1./np.random.chisquare(df=self.nu_0)*self.nu_0*self.sigsqr_0
            ret = np.zeros(1, dtype=self.model_dtype)
            ret['mu'] = np.random.normal(self.mu_0, np.sqrt(var/self.kappa_0))
            ret['var'] = var
            return ret[0]
        else:
            var = 1./np.random.chisquare(df=self.nu_0, size=size)*self.nu_0*self.sigsqr_0
            ret = np.zeros(size, dtype=self.model_dtype)
            ret['mu'] = (np.random.normal(self.mu_0, np.sqrt(1./self.kappa_0), size=size)
                         * np.sqrt(var))
            ret['var'] = var
            return ret

    def like1(self, *args):
        """Returns likelihood Pr(x | mu, var), for a single data point."""
        if len(args) == 3:
            x, mu, var = args
        elif len(args) == 2:
            x, theta = args
            mu = theta['mu']
            var = theta['var']
        return np.exp(-0.5*(x-mu)**2/var) / np.sqrt(2*np.pi*var)

    def __call__(self, *args):
        """Returns Pr(mu, var), i.e., the prior density."""
        if len(args) == 2:
            mu, var = args
        elif len(args) == 1:
            mu = args[0]['mu']
            var = args[0]['var']
        return (normal_density(self.mu_0, var/self.kappa_0, mu) *
                scaled_IX_density(self.nu_0, self.sigsqr_0, var))

    def _post_params(self, D):
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
        mu_n, kappa_n, sigsqr_n, nu_n = self._post_params(D)
        try:
            n = len(D)
        except:
            n = 1
        return (gamma(nu_n/2.0)/gamma(self.nu_0/2.0) * np.sqrt(self.kappa_0/kappa_n) *
                (self.nu_0*self.sigsqr_0)**(self.nu_0/2.0) /
                (nu_n*sigsqr_n)**(nu_n/2.0) /
                np.pi**(n/2.0))

    def marginal_var(self, var):
        """Return Pr(var)"""
        return scaled_IX_density(self.nu_0, self.sigsqr_0, var)

    def marginal_mu(self, mu):
        return t_density(self.nu_0, self.mu_0, self.sigsqr_0/self.kappa_0, mu)


class NormInvGamma(Prior):
    """Normal-Inverse-Gamma prior for univariate Gaussian with params for mean and variance.

    Model parameters
    ----------------
    mu :    mean
    var :   variance

    Prior parameters
    ----------------
    mu_0 :  prior mean
    V_0
    a_0, b_0 : gamma parameters (note these are a/b-like, not alpha/beta-like)
    """
    def __init__(self, m_0, V_0, a_0, b_0):
        self.m_0 = float(m_0)
        self.V_0 = float(V_0)
        self.a_0 = float(a_0)
        self.b_0 = float(b_0)
        super(NormInvGamma, self).__init__()

    model_dtype = np.dtype([('mu', float), ('var', float)])

    def sample(self, size=None):
        if size is None:
            var = 1./np.random.gamma(self.a_0, scale=1./self.b_0)
            ret = np.zeros(1, dtype=self.model_dtype)
            ret['mu'] = np.random.normal(self.m_0, np.sqrt(self.V_0*var))
            ret['var'] = var
            return ret[0]
        else:
            var = 1./np.random.gamma(self.a_0, scale=1./self.b_0, size=size)
            ret = np.zeros(size, dtype=self.model_dtype)
            ret['mu'] = np.random.normal(self.m_0, np.sqrt(self.V_0), size=size)*np.sqrt(var)
            ret['var'] = var
            return ret

    def like1(self, *args):
        """Returns likelihood Pr(x | mu, var), for a single data point."""
        if len(args) == 3:
            x, mu, var = args
        elif len(args) == 2:
            x, theta = args
            mu = theta['mu']
            var = theta['var']
        return np.exp(-0.5*(x-mu)**2/var) / np.sqrt(2*np.pi*var)

    def __call__(self, *args):
        """Returns Pr(mu, var), i.e., the prior density."""
        if len(args) == 1:
            mu = args[0]['mu']
            var = args[0]['var']
        elif len(args) == 2:
            mu, var = args
        normal = np.exp(-0.5*(self.m_0-mu)**2/(var*self.V_0))/np.sqrt(2*np.pi*var*self.V_0)
        ig = self.b_0**self.a_0/gamma(self.a_0)*var**(-(self.a_0+1))*np.exp(-self.b_0/var)
        return normal*ig

    def _post_params(self, D):
        try:
            n = len(D)
        except TypeError:
            n = 1
        Dbar = np.mean(D)
        invV_0 = 1./self.V_0
        V_n = 1./(invV_0 + n)
        m_n = V_n*(invV_0*self.m_0 + n*Dbar)
        a_n = self.a_0 + n/2.0
        # The commented line below is from Murphy.  It doesn't pass the unit tests so I derived
        # my own formula which does.
        # b_n = self.b_0 + 0.5*(self.m_0**2*invV_0 + np.sum(Dbar**2) - m_n**2/V_n)
        b_n = self.b_0 + 0.5*(np.sum((D-Dbar)**2)+n/(1.0+n*self.V_0)*(self.m_0-Dbar)**2)
        return m_n, V_n, a_n, b_n

    def pred(self, x):
        """Prior predictive.  Pr(x)"""
        return t_density(2.0*self.a_0, self.m_0, self.b_0*(1.0+self.V_0)/self.a_0, x)

    def evidence(self, D):
        """Fully marginalized likelihood Pr(D)"""
        m_n, V_n, a_n, b_n = self._post_params(D)
        try:
            n = len(D)
        except:
            n = 1
        return (np.sqrt(np.abs(V_n/self.V_0)) * (self.b_0**self.a_0)/(b_n**a_n) *
                gamma(a_n)/gamma(self.a_0) / (np.pi**(n/2.0)*2.0**(n/2.0)))

    def marginal_var(self, var):
        """Return Pr(var)"""
        # Don't have an independent source for this, so convert params to NIX and use that result.
        nu_0 = 2*self.a_0
        sigsqr_0 = 2*self.b_0/nu_0
        return scaled_IX_density(nu_0, sigsqr_0, var)

    def marginal_mu(self, mu):
        """Return Pr(mu)"""
        # Don't have an independent source for this, so convert params to NIX and use that result.
        mu_0 = self.m_0
        kappa_0 = 1./self.V_0
        nu_0 = 2*self.a_0
        sigsqr_0 = 2*self.b_0/nu_0
        return t_density(nu_0, mu_0, sigsqr_0/kappa_0, mu)


class NormInvWish(Prior):
    """Normal-Inverse-Wishart prior for multivariate Gaussian distribution.

    Model parameters
    ----------------
    mu :    multivariate mean
    Sig : covariance matrix

    Prior parameters
    ----------------
    mu_0
    kappa_0
    Lam_0
    nu_0
    """
    def __init__(self, mu_0, kappa_0, Lam_0, nu_0):
        self.mu_0 = np.array(mu_0, dtype=float)
        self.kappa_0 = float(kappa_0)
        self.Lam_0 = np.array(Lam_0, dtype=float)
        self.nu_0 = int(nu_0)
        self.d = len(mu_0)
        self.model_dtype = np.dtype([('mu', float, self.d), ('Sig', float, (self.d, self.d))])
        super(NormInvWish, self).__init__()

    def _S(self, D):
        """Scatter matrix.  D is [NOBS, NDIM].  Returns [NDIM, NDIM] array."""
        # Eq (244)
        Dbar = np.mean(D, axis=0)
        return np.dot((D-Dbar).T, (D-Dbar))

    def sample(self, size=None):
        """Return a sample {mu, Sig} or list of samples [{mu_1, Sig_1}, ...] from
        distribution.
        """
        Sig = random_invwish(dof=self.nu_0, invS=self.Lam_0, size=size)
        if size is None:
            ret = np.zeros(1, dtype=self.model_dtype)
            ret['Sig'] = Sig
            ret['mu'] = np.random.multivariate_normal(self.mu_0, Sig/self.kappa_0)
            return ret[0]
        else:
            ret = np.zeros(size, dtype=self.model_dtype)
            ret['Sig'] = Sig
            for r in ret.ravel():
                r['mu'] = np.random.multivariate_normal(self.mu_0, r['Sig']/self.kappa_0)
            return ret

    def like1(self, *args):
        """Returns likelihood Pr(x | mu, Sig), for a single data point."""
        if len(args) == 2:
            x, theta = args
            mu = theta['mu']
            Sig = theta['Sig']
        elif len(args) == 3:
            x, mu, Sig = args
        assert x.shape[-1] == self.d
        assert mu.shape[-1] == self.d
        assert Sig.shape[-1] == Sig.shape[-2] == self.d
        norm = np.sqrt((2*np.pi)**self.d * np.linalg.det(Sig))
        # Tricky to make this broadcastable...
        einsum = np.einsum("...i,...ij,...j", x-mu, np.linalg.inv(Sig), x-mu)
        return np.exp(-0.5*einsum)/norm

    def __call__(self, *args):
        """Returns Pr(mu, Sig), i.e., the prior."""
        if len(args) == 1:
            mu = args[0]['mu']
            Sig = args[0]['Sig']
        elif len(args) == 2:
            mu, Sig = args
        nu_0, d = self.nu_0, self.d
        # Eq (249)
        Z = (2.0**(nu_0*d/2.0) * gammad(d, nu_0/2.0) *
             (2.0*np.pi/self.kappa_0)**(d/2.0) / np.linalg.det(self.Lam_0)**(nu_0/2.0))
        detSig = np.linalg.det(Sig)
        invSig = np.linalg.inv(Sig)
        einsum = np.einsum("...i,...ij,...j", mu-self.mu_0, invSig, mu-self.mu_0)
        # Eq (248)
        return 1./Z * detSig**(-((nu_0+d)/2.0+1.0)) * np.exp(
            -0.5*np.trace(np.einsum("...ij,...jk->...ik", self.Lam_0, invSig), axis1=-2, axis2=-1) -
            self.kappa_0/2.0*einsum)
        # return 1./Z * detSig**(-((nu_0+d)/2.0+1.0)) * np.exp(
        #     -0.5*np.trace(np.dot(self.Lam_0, invSig)) -
        #     self.kappa_0/2.0*einsum)

    def _post_params(self, D):
        """Recall D is [NOBS, NDIM]."""
        shape = D.shape
        if len(shape) == 2:
            n = shape[0]
            Dbar = np.mean(D, axis=0)
        elif len(shape) == 1:
            n = 1
            Dbar = np.mean(D)
        # Eq (252)
        kappa_n = self.kappa_0 + n
        # Eq (253)
        nu_n = self.nu_0 + n
        # Eq (251) (note typo in original, mu+0 -> mu_0)
        mu_n = (self.kappa_0 * self.mu_0 + n * Dbar) / kappa_n
        # Eq (254)
        x = (Dbar-self.mu_0)[:, np.newaxis]
        Lam_n = (self.Lam_0 +
                 self._S(D) +
                 self.kappa_0*n/kappa_n*np.dot(x, x.T))
        return mu_n, kappa_n, Lam_n, nu_n

    def pred(self, x):
        """Prior predictive.  Pr(x)"""
        return multivariate_t_density(self.nu_0-self.d+1, self.mu_0,
                                      self.Lam_0*(self.kappa_0+1)/(self.kappa_0 - self.d + 1), x)

    def evidence(self, D):
        """Return Pr(D) = \int Pr(D | theta) Pr(theta)"""
        shape = D.shape
        if len(shape) == 2:
            n, d = shape
        elif len(shape) == 1:
            n, d = 1, shape[0]
        assert d == self.d
        # Eq (266)
        mu_n, kappa_n, Lam_n, nu_n = self._post_params(D)
        detLam0 = np.linalg.det(self.Lam_0)
        detLamn = np.linalg.det(Lam_n)
        num = gammad(d, nu_n/2.0) * detLam0**(self.nu_0/2.0)
        den = np.pi**(n*d/2.0) * gammad(d, self.nu_0/2.0) * detLamn**(nu_n/2.0)
        return num/den * (self.kappa_0/kappa_n)**(d/2.0)
