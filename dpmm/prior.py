# Equation numbers refer to Kevin Murphy's "Conjugate Bayesian analysis of the Gaussian
# distribution" note unless otherwise specified.

from operator import mul
import numpy as np
from scipy.special import gamma
from utils import vTmv, gammad, random_invwish, pick_discrete
from density import multivariate_t_density, t_density, normal_density, scaled_IX_density
from data import PseudoMarginalData


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

    def _like1(x, *args, **kwargs):
        raise NotImplementedError

    def like1(self, x, *args, **kwargs):
        """Return likelihood for single data element.  Pr(x | theta)"""
        if isinstance(x, PseudoMarginalData):
            return np.sum(self._like1(x.data, *args, **kwargs) /
                          x.interim_prior[..., np.newaxis]) / x.nsample
        else:
            return self._like1(x, *args, **kwargs)

    def likelihood(self, D, *args, **kwargs):
        # It's quite likely overriding this will yield faster results...
        """Returns Pr(D | theta)."""
        return reduce(mul, (self.like1(x, *args, **kwargs) for x in D), 1.0)

    def __call__(self, *args):
        """Returns Pr(theta), i.e. the prior probability."""
        raise NotImplementedError

    def _post_params(self, D):
        raise NotImplementedError

    def post_params(self, D):
        """Returns new parameters for updating prior->posterior by initializing new object."""
        if isinstance(D, PseudoMarginalData):
            return self._post_params(D.random_sample())
        else:
            return self._post_params(D)

    def post(self, D):
        """Returns new Prior with updated params for prior->posterior."""
        return self._post(*self.post_params(D))

    def post_sample(self, D, *args, **kwargs):
        if isinstance(D, PseudoMarginalData):
            return self._post_sample(D, *args, **kwargs)
        else:
            return self.post(D).sample(*args, **kwargs)

    def _pred(self, x):
        raise NotImplementedError

    def pred(self, x):
        """Prior predictive.  Pr(x | params)"""
        if isinstance(x, PseudoMarginalData):
            return np.sum(self._pred(x.data)/x.interim_prior[..., np.newaxis], axis=(1, 2))/x.nsample
        else:
            return self._pred(x)


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
        self.nu_0 = nu_0
        self.d = len(mu_0)
        super(NormInvWish, self).__init__()

    def _S(self, D):
        """Scatter matrix.  D is [NOBS, NDIM].  Returns [NDIM, NDIM] array."""
        # Eq (244)
        Dbar = np.mean(D, axis=0)
        return vTmv(D-Dbar)

    def sample(self, size=1):
        """Return a sample {mu, Sig} or list of samples [{mu_1, Sig_1}, ...] from
        distribution.
        """
        Sig = random_invwish(dof=self.nu_0, invS=self.Lam_0, size=size)
        if size == 1:
            return np.random.multivariate_normal(self.mu_0, Sig/self.kappa_0), Sig
        else:
            return zip((np.random.multivariate_normal(self.mu_0, S/self.kappa_0) for S in Sig),
                       Sig)

    def _like1(self, x, mu, Sig):
        """Returns likelihood Pr(x | mu, Sig), for a single data point."""
        norm = np.sqrt((2*np.pi)**self.d * np.linalg.det(Sig))
        return np.exp(-0.5*vTmv(x-mu, np.linalg.inv(Sig)).flat[0]) / norm

    def __call__(self, mu, Sig):
        """Returns Pr(mu, Sig), i.e., the prior."""
        nu_0, d = self.nu_0, self.d
        # Eq (249)
        Z = (2.0**(nu_0*d/2.0) * gammad(d, nu_0/2.0) *
             (2.0*np.pi/self.kappa_0)**(d/2.0) / np.linalg.det(self.Lam_0)**(nu_0/2.0))
        detSig = np.linalg.det(Sig)
        invSig = np.linalg.inv(Sig)
        # Eq (248)
        return 1./Z * detSig**(-((nu_0+d)/2.0+1.0)) * np.exp(
            -0.5*np.trace(np.dot(self.Lam_0, invSig)) -
            self.kappa_0/2.0*vTmv(mu-self.mu_0, invSig).flat[0])

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
                 self.kappa_0*n/kappa_n*vTmv(x.T))
        return mu_n, kappa_n, Lam_n, nu_n

    def _pred(self, x):
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
        mu_n, kappa_n, Lam_n, nu_n = self.post_params(D)
        detLam0 = np.linalg.det(self.Lam_0)
        detLamn = np.linalg.det(Lam_n)
        num = gammad(d, nu_n/2.0) * detLam0**(self.nu_0/2.0)
        den = np.pi**(n*d/2.0) * gammad(d, self.nu_0/2.0) * detLamn**(nu_n/2.0)
        return num/den * (self.kappa_0/kappa_n)**(d/2.0)


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
    sigsqr : Known variance.  Treat as a prior parameter to make __init__() with with post_params(),
             post(), post_pred(), etc., though note this never actually gets updated.
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
            return (np.random.normal(self.mu_0, np.sqrt(self.sigsqr_0)),)
        else:
            return np.random.normal(self.mu_0, np.sqrt(self.sigsqr_0), size=size)

    def _like1(self, x, mu):
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

    def _pred(self, x):
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

    def sample(self, size=1):
        # sanity check this.
        var = 1./np.random.chisquare(df=self.nu_0, size=size)*self.nu_0*self.sigsqr_0
        if size == 1:
            return np.random.normal(self.mu_0, np.sqrt(var/self.kappa_0)), var
        else:
            return zip((np.random.normal(self.mu_0, np.sqrt(v/self.kappa_0)) for v in var), var)

    def _post_sample(self, D, current_mu, current_var):
        # D here is a PseudoMarginalData with NDIM=1.
        # Trying a MH update, so need the current posterior and a proposal posterior
        # For simplicity, also want the proposal transition probability to be symmetric.
        std_sample_mean = np.std(np.mean(D.data[..., 0]/D.interim_prior[:, 0], axis=1))
        proposed_mu = current_mu + np.random.normal(scale=std_sample_mean/np.sqrt(len(D)))
        proposed_var = current_var + np.random.normal(scale=std_sample_mean/np.sqrt(2*len(D)))
        current_pr = self.likelihood(D, (current_mu, current_var))*self(current_mu, proposed_mu)
        proposed_pr = self.likelihood(D, (proposed_mu, proposed_var))*self(proposed_mu, proposed_var)
        if proposed_pr > current_pr:
            return (proposed_mu, proposed_var)
        elif np.random.rand() < proposed_pr/current_pr:
            return (proposed_mu, proposed_var)
        else:
            return current_mu, current_var

    def _like1(self, x, mu, var):
        """Returns likelihood Pr(x | mu, var), for a single data point."""
        return np.exp(-0.5*(x-mu)**2/var) / np.sqrt(2*np.pi*var)

    def __call__(self, mu, var):
        """Returns Pr(mu, var), i.e., the prior density."""
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

    def _pred(self, x):
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

    def sample(self, size=1):
        var = 1./np.random.gamma(self.a_0, scale=1./self.b_0, size=size)
        if size == 1:
            return np.random.normal(self.m_0, np.sqrt(self.V_0*var)), var
        else:
            return zip(np.random.normal(self.m_0, np.sqrt(self.V_0*var), size=size), var)

    def _like1(self, x, mu, var):
        """Returns likelihood Pr(x | mu, var), for a single data point."""
        if isinstance(x, PseudoMarginalData):
            return np.sum(self.like1(x.data, mu, var)/x.interim_post) / x.nsample
        else:
            return np.exp(-0.5*(x-mu)**2/var) / np.sqrt(2*np.pi*var)

    def __call__(self, mu, var):
        """Returns Pr(mu, var), i.e., the prior density."""
        normal = np.exp(-0.5*(self.m_0-mu)**2/(var*self.V_0))/np.sqrt(2*np.pi*var*self.V_0)
        ig = self.b_0**self.a_0/gamma(self.a_0)*var**(-(self.a_0+1))*np.exp(-self.b_0/var)
        return normal*ig

    def _post_params(self, D):
        if isinstance(D, PseudoMarginalData):
            pass
        else:
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

    def _pred(self, x):
        """Prior predictive.  Pr(x)"""
        if isinstance(x, PseudoMarginalData):
            return np.sum(self.pred(x.data)/x.interim_post) / x.nsample
        else:
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


class InvGamma(Prior):
    """Inverse Gamma distribution.  Note this parameterization matches Murphy's, not wikipedia's.s"""
    def __init__(self, alpha, beta, mu):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        super(InvGamma, self).__init__()

    def sample(self, size=1):
        return 1./np.random.gamma(self.alpha, scale=self.beta, size=size)

    def _like1(self, x, var):
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

    def _pred(self, x):
        """Prior predictive.  Pr(x)"""
        return t_density(2*self.alpha, self.mu, self.beta/self.alpha, x)

    def evidence(self, D):
        """Fully marginalized likelihood Pr(D)"""
        raise NotImplementedError


class InvWish(Prior):
    """Under construction."""
    def __init__(self, nu, Psi, mu):
        self.nu = int(nu)
        self.Psi = Psi
        self.mu = mu
        self.p = len(self.mu)
        super(InvWish, self).__init__()

    def _S(self, D):
        """Scatter matrix.  D is [NOBS, NDIM].  Returns [NDIM, NDIM] array."""
        # Eq (244)
        Dbar = np.mean(D, axis=0)
        return vTmv(D-Dbar)

    def sample(self, size=1):
        Sig = random_invwish(dof=self.nu, invS=self.Psi, size=size)

    def _like1(self, x, Sig):
        norm = np.sqrt((2*np.pi)**self.p * np.linalg.det(Sig))
        return np.exp(-0.5*vTmv(x-self.mu, np.linalg.inv(Sig)).flat[0]) / norm

    def __call__(self, Sig):
        nu, p = self.nu, self.p
        detPsi = np.linalg.det(self.Psi)
        invPsi = np.linalg.inv(self.Psi)
        detSig = np.linalg.det(Sig)
        invSig = np.linalg.inv(Sig)
        return (detPsi**(nu/2.0) / (2.**(nu*p/2.0) * gammad(p, nu/2.0))*detSig**(-(nu+p+1.0)/2.0) *
                np.exp(-0.5*np.trace(np.dot(self.Psi, invSig))))

    def _post_params(self, D):
        shape = D.shape
        if len(shape) == 2:
            n, d = shape
            Dbar = np.mean(D, axis=0)
        elif len(shape) == 1:
            n, d = 1, shape[0]
            Dbar = np.mean(D)
        nu_n = self.nu + n
        Psi_n = self.Psi + vTmv((D-self.mu))
        return nu_n, Psi_n, self.mu

    def _pred(self, x):
        return multivariate_t_density(self.nu-self.p+1, self.mu, self.Psi/(self.nu-self.p+1), x)
