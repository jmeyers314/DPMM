import numpy as np
from scipy.integrate import quad, dblquad

import prior


def test_NIW():
    mu_0 = np.r_[0.0, 0.0]
    kappa_0 = 2
    Lam_0 = np.eye(2)
    nu_0 = 2

    # Create a Normal-Inverse-Wishart prior.
    niw = prior.NIW(mu_0, kappa_0, Lam_0, nu_0)

    # Check that we can draw samples from niw.
    niw.sample()
    niw.sample(size=10)

    # Check that we can evaluate a likelihood given 1 data point.
    theta = (np.r_[1., 1.], np.eye(2)+0.12)
    x = np.r_[0.1, 0.2]
    niw.like1(*theta, x=x)
    # Or given multiple data points.
    D = np.array([[0.1, 0.2], [0.2, 0.3], [0.1, 0.2], [0.4, 0.3]])
    print niw.likelihood(*theta, D=D)

    # Evaluate prior
    print niw(*theta)
    print niw.post_params(D)
    print niw.pred(x)
    print niw.post_pred(D, x)


def test_GaussianMeanKnownVariance():
    mu_0 = 0.0
    sig_0 = 1.0
    sig = 0.1
    model = prior.GaussianMeanKnownVariance(mu_0, sig_0, sig)

    # Check that we can draw samples from model.
    model.sample()
    model.sample(size=10)

    # Check that we can evaluate a likelihood given 1 data point.
    theta = (1.0, )
    x = 1.0
    model.like1(*theta, x=x)
    # Or given multiple data points.
    D = np.array([1.0, 1.0, 1.0, 1.3])
    print model.likelihood(*theta, D=D)

    # Evaluate prior
    print model(*theta)
    # Update prior parameters
    print model.post_params(D)
    # Prior predictive
    print model.pred(x)
    # Posterior predictive
    print model.post_pred(D, x)


def test_NIG():
    mu_0 = 0.0
    V_0 = 1
    a_0 = 1.0
    b_0 = 1.0
    model = prior.NIG(mu_0, V_0, a_0, b_0)

    # Check that we can draw samples from model.
    model.sample()
    model.sample(size=10)

    # Check that we can evaluate a likelihood given 1 data point.
    theta = (1.0, 1.0)
    x = 0.5
    model.like1(*theta, x=x)
    # Or given multiple data points.
    D = np.array([1.0, 1.0, 1.0, 1.3])
    print model.likelihood(*theta, D=D)

    # Evaluate prior
    print model(*theta)
    # Update prior parameters
    print model.post_params(D)
    # Prior predictive
    print model.pred(x)
    # Posterior predictive
    print model.post_pred(D, x)


def test_NIX_eq_NIG():
    mu_0 = 0.1
    sigsqr_0 = 1.1
    kappa_0 = 2
    nu_0 = 3

    m_0 = mu_0
    V_0 = 1./kappa_0
    a_0 = nu_0/2.0
    b_0 = nu_0*sigsqr_0/2.0

    model1 = prior.NIX(mu_0, kappa_0, sigsqr_0, nu_0)
    model2 = prior.NIG(m_0, V_0, a_0, b_0)

    mus = np.linspace(-2.2, 2.2, 5)
    vars_ = np.linspace(1.0, 4.0, 5)
    xs = np.arange(-1.1, 1.1, 5)

    for x in xs:
        np.testing.assert_equal(
            model1.pred(x), model2.pred(x),
            "NIX and NIG prior predictive densities don't agree at x = ".format(x))
        np.testing.assert_equal(
            model1.post(x).pred(x), model2.post(x).pred(x),
            "NIX and NIG posterior predictive densities don't agree at x = {}".format(x))

    for mu, var in zip(mus, vars_):
        np.testing.assert_almost_equal(
            model1(mu, var), model2(mu, var), 10,
            "NIX and NIG prior densities don't agree at mu, var = {}, {}".format(mu, var))

    post1 = model1.post(xs)
    post2 = model2.post(xs)
    for mu, var in zip(mus, vars_):
        np.testing.assert_almost_equal(
            post1(mu, var), post2(mu, var), 10,
            "NIX and NIG posterior densities don't agree at mu, var = {}, {}".format(mu, var))

    for mu, var, x in zip(mus, vars_, xs):
        np.testing.assert_almost_equal(
            model1.like1(mu, var, x), model2.like1(mu, var, x), 10,
            "NIX and NIG likelihoods don't agree at mu, var, x = {}, {}, {}".format(mu, var, x))

    np.testing.assert_almost_equal(
        model1.evidence(xs), model2.evidence(xs), 10,
        "NIX and NIG evidences don't agree")


def test_NIX_integrate():
    import warnings
    mu_0 = -0.1
    sigsqr_0 = 1.1
    kappa_0 = 2
    nu_0 = 3

    nix = prior.NIX(mu_0, kappa_0, sigsqr_0, nu_0)

    # Check prior density
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = dblquad(nix, 0.0, np.inf, lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 5, "NIX prior density does not integrate to 1.0")

    # Check prior predictive density
    r = quad(nix.pred, -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "NIX prior predictive density does not integrate to 1.0")

    # Check posterior density
    D = [1.0, 2.0, 3.0]
    r = dblquad(nix.post(D), 0.0, np.inf, lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 7, "NIX posterior density does not integrate to 1.0")

    # Check posterior predictive density
    r = quad(nix.post(D).pred, -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "NIX posterior predictive density does not integrate to 1.0")

    # Check that the likelihood integrates to 1.
    r = quad(lambda x: nix.like1(mu=1.1, var=2.1, x=x), -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "NIX likelihood does not integrate to 1.0")

    # Check that evidence (of single data point) integrates to 1.
    r = quad(lambda x: nix.evidence(x), -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "NIX evidence does not integrate to 1.0")
    # Check evidence for two data points.
    r = dblquad(lambda x, y: nix.evidence([x, y]),
                -np.inf, np.inf,
                lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 5,
                                   "NIX evidence does not integrate to 1.0")

    # Check that posterior = prior * likelihood / evidence
    mus = [1.1, 1.2, 1.3]
    vars_ = [1.2, 3.2, 2.3]
    post = nix.post(D)
    post1 = [nix(mu, var)*nix.likelihood(mu, var, D=D) / nix.evidence(D)
             for mu, var in zip(mus, vars_)]
    post2 = [post(mu, var) for mu, var in zip(mus, vars_)]
    np.testing.assert_array_almost_equal(post1, post2, 10,
                                         "NIX posterior != prior * likelihood / evidence")


def test_NIG_integrate():
    import warnings
    m_0 = -0.1
    V_0 = 1.1
    a_0 = 2.0
    b_0 = 3.0

    nig = prior.NIG(m_0, V_0, a_0, b_0)

    # Check prior density
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = dblquad(nig, 0.0, np.inf, lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 5, "NIG prior density does not integrate to 1.0")

    # Check prior predictive density
    r = quad(nig.pred, -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "NIG prior predictive density does not integrate to 1.0")

    # Check posterior density
    D = [1.0, 2.0, 3.0]
    r = dblquad(nig.post(D), 0.0, np.inf, lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 7, "NIG posterior density does not integrate to 1.0")

    # Check posterior predictive density
    r = quad(nig.post(D).pred, -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "NIG posterior predictive density does not integrate to 1.0")

    # Check that the likelihood integrates to 1.  (Should it?)
    r = quad(lambda x: nig.like1(mu=1.1, var=2.1, x=x), -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "NIG likelihood does not integrate to 1.0")

    # Check that evidence (of single data point) integrates to 1.
    r = quad(lambda x: nig.evidence(x), -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "NIG evidence does not integrate to 1.0")
    # Check evidence for two data points.
    r = dblquad(lambda x, y: nig.evidence([x, y]),
                -np.inf, np.inf,
                lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 5,
                                   "NIG evidence does not integrate to 1.0")

    # Check that posterior = prior * likelihood / evidence
    mus = [1.1, 1.2, 1.3]
    vars_ = [1.2, 3.2, 2.3]
    post = nig.post(D)
    post1 = [nig(mu, var)*nig.likelihood(mu, var, D=D) / nig.evidence(D)
             for mu, var in zip(mus, vars_)]
    post2 = [post(mu, var) for mu, var in zip(mus, vars_)]
    np.testing.assert_array_almost_equal(post1, post2, 10,
                                         "NIG posterior != prior * likelihood / evidence")


def test_scaled_IX_density():
    nu = 1
    sigsqr = 1.0
    r = quad(lambda x: prior.scaled_IX_density(nu, sigsqr, x), 0.0, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10, "scaled_IX_density does not integrate to 1.0")


if __name__ == "__main__":
    test_NIW()
    test_GaussianMeanKnownVariance()
    test_NIG()
    test_NIX_eq_NIG()
    test_NIX_integrate()
    test_NIG_integrate()
    test_scaled_IX_density()
