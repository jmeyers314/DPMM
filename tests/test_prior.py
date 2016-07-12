import warnings
import numpy as np
from scipy.integrate import quad, dblquad, tplquad

import dpmm
from test_utils import timer


@timer
def test_GaussianMeanKnownVariance():
    mu_0 = 0.15
    sigsqr_0 = 1.2
    sigsqr = 0.15
    model = dpmm.GaussianMeanKnownVariance(mu_0, sigsqr_0, sigsqr)

    D = np.r_[1.0, 2.2, 1.1, -1.13]
    mus = np.r_[1.1, 2.0, 0.1]

    # Check prior density
    r = quad(model, -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "GaussianMeanKnownVariance prior density does not integrate to 1.0")

    # Check prior predictive density
    r = quad(model.pred, -np.inf, np.inf)
    np.testing.assert_almost_equal(
        r[0], 1.0, 10,
        "GaussianMeanKnownVariance prior predictive density does not integrate to 1.0")

    # Check posterior density
    r = quad(model.post(D), -np.inf, np.inf)
    np.testing.assert_almost_equal(
        r[0], 1.0, 10,
        "GaussianMeanKnownVariance posterior density does not integrate to 1.0")

    # Check posterior predictive density
    r = quad(model.post(D).pred, -np.inf, np.inf)
    np.testing.assert_almost_equal(
        r[0], 1.0, 10,
        "GaussianMeanKnownVariance posterior predictive density does not integrate to 1.0")

    # Check that the likelihood integrates to 1.
    r = quad(lambda x: model.like1(x, mu=1.1), -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "GaussianMeanKnownVariance likelihood does not integrate to 1.0")

    # # Check that evidence (of single data point) integrates to 1.
    # r = quad(lambda x: model.evidence(x), -np.inf, np.inf)
    # np.testing.assert_almost_equal(r[0], 1.0, 10,
    #                                "GaussianMeanKnownVariance evidence does not integrate to 1.0")

    # # Check evidence for two data points.
    # r = dblquad(lambda x, y: model.evidence([x, y]),
    #             -np.inf, np.inf,
    #             lambda x: -np.inf, lambda x: np.inf)
    # np.testing.assert_almost_equal(r[0], 1.0, 5,
    #                                "GaussianMeanKnownVariance evidence does not integrate to 1.0")

    # # Check that posterior = prior * likelihood / evidence
    # post = model.post(D)
    # post1 = [model(mu)*model.likelihood(mu, D=D) / model.evidence(D) for mu in mus]
    # post2 = [post(mu) for mu in mus]
    # np.testing.assert_array_almost_equal(
    #     post1, post2, 10,
    #     "GaussianMeanKnownVariance posterior != prior * likelihood / evidence")

    # Check that posterior is proportional to prior * likelihood
    # Add some more data points
    posts = [model.post(D)(mu) for mu in mus]
    posts2 = [model(mu)*model.likelihood(D, mu) for mu in mus]

    np.testing.assert_array_almost_equal(
        posts/posts[0], posts2/posts2[0], 5,
        "GaussianMeanKnownVariance posterior not proportional to prior * likelihood.")

    # Check that integrating out theta yields the prior predictive.
    xs = [0.1, 0.2, 0.3, 0.4]
    preds1 = np.array([quad(lambda theta: model(theta) * model.like1(x, theta), -np.inf, np.inf)[0] for x in xs])
    preds2 = np.array([model.pred(x) for x in xs])

    np.testing.assert_array_almost_equal(
            preds1/preds1[0], preds2/preds2[0], 5,
            "Prior predictive not proportional to integral of likelihood * prior")

@timer
def test_InvGamma():
    alpha = 1.1
    beta = 1.2
    mu = 0.1
    ig = dpmm.InvGamma(alpha, beta, mu)
    ig.sample()

    # Check prior density
    r = quad(ig, 0.0, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 5, "InvGamma prior density does not integrate to 1.0")

    # Check prior predictive density
    r = quad(ig.pred, -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "InvGamma prior predictive density does not integrate to 1.0")

    # Check posterior density
    D = [1.0, 2.0, 3.0]
    r = quad(ig.post(D), 0.0, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 7,
                                   "InvGamma posterior density does not integrate to 1.0")

    # Check posterior predictive density
    r = quad(ig.post(D).pred, -np.inf, np.inf)
    np.testing.assert_almost_equal(
        r[0], 1.0, 10, "InvGamma posterior predictive density does not integrate to 1.0")

    # Check that the likelihood integrates to 1.
    r = quad(lambda x: ig.like1(x, var=2.1), -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "InvGamma likelihood does not integrate to 1.0")

    # Check that posterior is proportional to prior * likelihood
    # Add some more data points
    D = np.array([1.0, 2.0, 3.0, 2.2, 2.3, 1.2])
    vars_ = [0.7, 1.1, 1.2, 1.5]
    posts = [ig.post(D)(var) for var in vars_]
    posts2 = [ig(var)*ig.likelihood(D, var) for var in vars_]

    np.testing.assert_array_almost_equal(
        posts/posts[0], posts2/posts2[0], 5,
        "InvGamma posterior not proportional to prior * likelihood.")

    # Check mean and variance
    mean = 1./beta/(alpha-1.0)
    np.testing.assert_almost_equal(quad(lambda x: ig(x)*x, 0.0, np.inf)[0], mean, 10,
                                   "InvGamma has wrong mean.")
    var = beta**(-2)/(alpha-1)**2/(alpha-2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        np.testing.assert_almost_equal(quad(lambda x: ig(x)*(x-mean)**2, 0.0, np.inf)[0], var, 5,
                                       "InvGamma has wrong variance.")

    # Check that integrating out theta yields the prior predictive.
    xs = [0.1, 0.2, 0.3, 0.4]
    preds1 = np.array([quad(lambda theta: ig(theta) * ig.like1(x, theta), 0, np.inf)[0] for x in xs])
    preds2 = np.array([ig.pred(x) for x in xs])

    np.testing.assert_array_almost_equal(
            preds1/preds1[0], preds2/preds2[0], 5,
            "Prior predictive not proportional to integral of likelihood * prior")


@timer
def test_InvGamma2D(full=False):
    alpha = 1.1
    beta = 1.2
    mu = np.r_[0.1, 0.2]
    ig2d = dpmm.InvGamma2D(alpha, beta, mu)
    ig2d.sample()

    # Check prior density
    r = quad(ig2d, 0.0, np.inf)
    np.testing.assert_almost_equal(
            r[0], 1.0, 5, "InvGamma2D prior density does not integrate to 1.0")

    if __name__ == '__main__' and full:
        # Check prior predictive density
        r = dblquad(lambda x, y: ig2d.pred(np.r_[x, y]),
                                           -np.inf, np.inf,
                                           lambda x: -np.inf, lambda x: np.inf)
        np.testing.assert_almost_equal(
                r[0], 1.0, 5, "InvGamma2D prior predictive density does not integrate to 1.0")

    # Check posterior density
    D = np.array([[0.1, 0.2], [0.2, 0.3]])
    r = quad(ig2d.post(D), 0.0, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 7,
                                   "InvGamma2D posterior density does not integrate to 1.0")

    # Check posterior predictive density
    r = dblquad(lambda x, y: ig2d.post(D).pred(np.r_[x, y]),
                -np.inf, np.inf,
                lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(
        r[0], 1.0, 5, "InvGamma2D posterior predictive density does not integrate to 1.0")

    # Check that the likelihood integrates to 1.
    r = dblquad(lambda x, y: ig2d.like1(np.r_[x, y], var=2.1),
                -np.inf, np.inf,
                lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "InvGamma2D likelihood does not integrate to 1.0")

    # Check that posterior is proportional to prior * likelihood
    vars_ = [0.7, 1.1, 1.2, 1.5]
    posts = np.array([ig2d.post(D)(var) for var in vars_])
    posts2 = np.array([ig2d(var)*ig2d.likelihood(D, var) for var in vars_])

    np.testing.assert_array_almost_equal(
        posts/posts[0], posts2/posts2[0], 5,
        "InvGamma2D posterior not proportional to prior * likelihood.")

    # Check mean and variance
    mean = 1./beta/(alpha-1.0)
    np.testing.assert_almost_equal(quad(lambda x: ig2d(x)*x, 0.0, np.inf)[0], mean, 10,
                                   "InvGamma2D has wrong mean.")
    var = beta**(-2)/(alpha-1)**2/(alpha-2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        np.testing.assert_almost_equal(quad(lambda x: ig2d(x)*(x-mean)**2, 0.0, np.inf)[0], var, 5,
                                       "InvGamma2D has wrong variance.")

    # Check that integrating out theta yields the prior predictive.
    xs = [np.r_[0.1, 0.2], np.r_[0.2, 0.3], np.r_[0.1, 0.3]]
    preds1 = np.array([quad(lambda theta: ig2d(theta) * ig2d.like1(x, theta), 0, np.inf)[0] for x in xs])
    preds2 = np.array([ig2d.pred(x) for x in xs])

    np.testing.assert_array_almost_equal(
             preds1/preds1[0], preds2/preds2[0], 5,
             "Prior predictive not proportional to integral of likelihood * prior")


@timer
def test_NormInvChi2():
    mu_0 = -0.1
    sigsqr_0 = 1.1
    kappa_0 = 2
    nu_0 = 3

    nix = dpmm.NormInvChi2(mu_0, kappa_0, sigsqr_0, nu_0)

    D = np.r_[1.0, 2.0, 3.0]
    mus = np.r_[1.1, 1.2, 1.3]
    vars_ = np.r_[1.2, 3.2, 2.3]

    # Check prior density
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = dblquad(nix, 0.0, np.inf, lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 5,
                                   "NormInvChi2 prior density does not integrate to 1.0")

    # Check prior predictive density
    r = quad(nix.pred, -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "NormInvChi2 prior predictive density does not integrate to 1.0")

    # Check posterior density
    r = dblquad(nix.post(D), 0.0, np.inf, lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 7,
                                   "NormInvChi2 posterior density does not integrate to 1.0")

    # Check posterior predictive density
    r = quad(nix.post(D).pred, -np.inf, np.inf)
    np.testing.assert_almost_equal(
        r[0], 1.0, 10,
        "NormInvChi2 posterior predictive density does not integrate to 1.0")

    # Check that the likelihood integrates to 1.
    r = quad(lambda x: nix.like1(x, 1.1, 2.1), -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "NormInvChi2 likelihood does not integrate to 1.0")

    # Check that evidence (of single data point) integrates to 1.
    r = quad(lambda x: nix.evidence(x), -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "NormInvChi2 evidence does not integrate to 1.0")
    # Check evidence for two data points.
    r = dblquad(lambda x, y: nix.evidence([x, y]),
                -np.inf, np.inf,
                lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 5,
                                   "NormInvChi2 evidence does not integrate to 1.0")

    # Check that posterior = prior * likelihood / evidence
    post = nix.post(D)
    post1 = [nix(mu, var)*nix.likelihood(D, mu, var) / nix.evidence(D)
             for mu, var in zip(mus, vars_)]
    post2 = [post(mu, var) for mu, var in zip(mus, vars_)]
    np.testing.assert_array_almost_equal(post1, post2, 10,
                                         "NormInvChi2 posterior != prior * likelihood / evidence")

    # Test that marginal variance probability method matches integrated result.
    Pr_var1 = [nix.marginal_var(var) for var in vars_]
    Pr_var2 = [quad(lambda mu: nix(mu, var), -np.inf, np.inf)[0] for var in vars_]
    np.testing.assert_array_almost_equal(
        Pr_var1, Pr_var2, 10,
        "Pr(var) method calculation does not match integrated result.")

    # Test that marginal mean probability method matches integrated result.
    Pr_mu1 = [nix.marginal_mu(mu) for mu in mus]
    Pr_mu2 = [quad(lambda var: nix(mu, var), 0.0, np.inf)[0] for mu in mus]
    np.testing.assert_array_almost_equal(
        Pr_mu1, Pr_mu2, 10,
        "Pr(mu) method calculation does not match integrated result.")

    # Check that integrating out theta yields the prior predictive.
    xs = [0.1, 0.2, 0.3, 0.4]
    preds1 = np.array([dblquad(lambda mu, var: nix(mu, var) * nix.like1(x, mu, var),
                               0, np.inf,
                               lambda var: -np.inf, lambda var: np.inf)[0]
                       for x in xs])
    preds2 = np.array([nix.pred(x) for x in xs])

    np.testing.assert_array_almost_equal(
         preds1/preds1[0], preds2/preds2[0], 5,
         "Prior predictive not proportional to integral of likelihood * prior")

@timer
def test_NormInvGamma():
    m_0 = -0.1
    V_0 = 1.1
    a_0 = 2.0
    b_0 = 3.0

    nig = dpmm.NormInvGamma(m_0, V_0, a_0, b_0)

    D = np.r_[1.0, 2.0, 3.0]
    mus = np.r_[1.1, 1.2, 1.3]
    vars_ = np.r_[1.2, 3.2, 2.3]

    # Check prior density
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = dblquad(nig, 0.0, np.inf, lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 5,
                                   "NormInvGamma prior density does not integrate to 1.0")

    # Check prior predictive density
    r = quad(nig.pred, -np.inf, np.inf)
    np.testing.assert_almost_equal(
        r[0], 1.0, 10,
        "NormInvGamma prior predictive density does not integrate to 1.0")

    # Check posterior density
    r = dblquad(nig.post(D), 0.0, np.inf, lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 7,
                                   "NormInvGamma posterior density does not integrate to 1.0")

    # Check posterior predictive density
    r = quad(nig.post(D).pred, -np.inf, np.inf)
    np.testing.assert_almost_equal(
        r[0], 1.0, 10,
        "NormInvGamma posterior predictive density does not integrate to 1.0")

    # Check that the likelihood integrates to 1.
    r = quad(lambda x: nig.like1(x, 1.1, 2.1), -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "NormInvGamma likelihood does not integrate to 1.0")

    # Check that evidence (of single data point) integrates to 1.
    r = quad(lambda x: nig.evidence(x), -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "NormInvGamma evidence does not integrate to 1.0")
    # Check evidence for two data points.
    r = dblquad(lambda x, y: nig.evidence([x, y]),
                -np.inf, np.inf,
                lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 5,
                                   "NormInvGamma evidence does not integrate to 1.0")

    # Check that posterior = prior * likelihood / evidence
    post = nig.post(D)
    post1 = [nig(mu, var)*nig.likelihood(D, mu, var) / nig.evidence(D)
             for mu, var in zip(mus, vars_)]
    post2 = [post(mu, var) for mu, var in zip(mus, vars_)]
    np.testing.assert_array_almost_equal(post1, post2, 10,
                                         "NormInvGamma posterior != prior * likelihood / evidence")

    # Test that marginal variance probability method matches integrated result.
    Pr_var1 = [nig.marginal_var(var) for var in vars_]
    Pr_var2 = [quad(lambda mu: nig(mu, var), -np.inf, np.inf)[0] for var in vars_]
    np.testing.assert_array_almost_equal(
        Pr_var1, Pr_var2, 10,
        "Pr(var) method calculation does not match integrated result.")

    # Test that marginal mean probability method matches integrated result.
    Pr_mu1 = [nig.marginal_mu(mu) for mu in mus]
    Pr_mu2 = [quad(lambda var: nig(mu, var), 0.0, np.inf)[0] for mu in mus]
    np.testing.assert_array_almost_equal(
        Pr_mu1, Pr_mu2, 10,
        "Pr(mu) method calculation does not match integrated result.")

    # Check that integrating out theta yields the prior predictive.
    xs = [0.1, 0.2, 0.3, 0.4]
    preds1 = np.array([dblquad(lambda mu, var: nig(mu, var) * nig.like1(x, mu, var),
                               0, np.inf,
                               lambda var: -np.inf, lambda var: np.inf)[0]
                       for x in xs])
    preds2 = np.array([nig.pred(x) for x in xs])

    np.testing.assert_array_almost_equal(
         preds1/preds1[0], preds2/preds2[0], 5,
         "Prior predictive not proportional to integral of likelihood * prior")

@timer
def test_NormInvChi2_eq_NormInvGamma():
    mu_0 = 0.1
    sigsqr_0 = 1.1
    kappa_0 = 2
    nu_0 = 3

    m_0 = mu_0
    V_0 = 1./kappa_0
    a_0 = nu_0/2.0
    b_0 = nu_0*sigsqr_0/2.0

    model1 = dpmm.NormInvChi2(mu_0, kappa_0, sigsqr_0, nu_0)
    model2 = dpmm.NormInvGamma(m_0, V_0, a_0, b_0)

    mus = np.linspace(-2.2, 2.2, 5)
    vars_ = np.linspace(1.0, 4.0, 5)
    xs = np.arange(-1.1, 1.1, 5)

    for x in xs:
        np.testing.assert_equal(
            model1.pred(x), model2.pred(x),
            "NormInvChi2 and NormInvGamma prior predictive densities don't agree at x = ".format(x))
        np.testing.assert_equal(
            model1.post(x).pred(x), model2.post(x).pred(x),
            "NormInvChi2 and NormInvGamma posterior " +
            "predictive densities don't agree at x = {}".format(x))

    for mu, var in zip(mus, vars_):
        np.testing.assert_almost_equal(
            model1(mu, var), model2(mu, var), 10,
            "NormInvChi2 and NormInvGamma prior densities " +
            "don't agree at mu, var = {}, {}".format(mu, var))

    post1 = model1.post(xs)
    post2 = model2.post(xs)
    for mu, var in zip(mus, vars_):
        np.testing.assert_almost_equal(
            post1(mu, var), post2(mu, var), 10,
            "NormInvChi2 and NormInvGamma posterior densities " +
            "don't agree at mu, var = {}, {}".format(mu, var))

    for mu, var, x in zip(mus, vars_, xs):
        np.testing.assert_almost_equal(
            model1.like1(x, mu, var), model2.like1(x, mu, var), 10,
            "NormInvChi2 and NormInvGamma likelihoods don't " +
            "agree at mu, var, x = {}, {}, {}".format(mu, var, x))

    np.testing.assert_almost_equal(
        model1.evidence(xs), model2.evidence(xs), 10,
        "NormInvChi2 and NormInvGamma evidences don't agree")


@timer
def test_NormInvWish(full=False):
    mu_0 = np.r_[0.2, 0.1]
    kappa_0 = 2.0
    Lam_0 = np.eye(2)+0.1
    nu_0 = 3

    # Create a Normal-Inverse-Wishart prior.
    niw = dpmm.NormInvWish(mu_0, kappa_0, Lam_0, nu_0)

    # Check that we can draw samples from NormInvWish.
    niw.sample()
    niw.sample(size=10)

    # Check that we can evaluate a likelihood given data.
    theta = np.zeros(1, dtype=niw.model_dtype)
    theta['mu'] = np.r_[1.0, 1.0]
    theta['Sig'] = np.eye(2)+0.12
    D = np.array([[0.1, 0.2], [0.2, 0.3], [0.1, 0.2], [0.4, 0.3]])
    niw.likelihood(D, theta)

    # Evaluate prior
    niw(theta)

    if __name__ == "__main__" and full:
        # Check prior predictive density
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = dblquad(lambda x, y: niw.pred(np.r_[x, y]), -np.inf, np.inf,
                        lambda x: -np.inf, lambda x: np.inf)
        np.testing.assert_almost_equal(r[0], 1.0, 5,
                                       "NormInvWish prior predictive density does not integrate to 1.0")

    # Check posterior predictive density
    r = dblquad(lambda x, y: niw.post(D).pred(np.r_[x, y]), -np.inf, np.inf,
                lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(
        r[0], 1.0, 5, "NormInvWish posterior predictive density does not integrate to 1.0")

    # Check that the likelihood of a single point in 2 dimensions integrates to 1.
    r = dblquad(lambda x, y: niw.like1(np.r_[x, y], np.r_[1.2, 1.1], np.eye(2)+0.12),
                -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10,
                                   "NormInvWish likelihood does not integrate to 1.0")

    if __name__ == "__main__" and full:
        # Check that likelihood of a single point in 3 dimensions integrates to 1.
        niw3 = dpmm.NormInvWish(np.r_[1, 1, 1], 2.0, np.eye(3), 3)
        r = tplquad(lambda x, y, z: niw3.like1(np.r_[x, y, z], np.r_[0.1, 0.2, 0.3], np.eye(3)+0.1),
                    -np.inf, np.inf,
                    lambda x: -np.inf, lambda x: np.inf,
                    lambda x, y: -np.inf, lambda x, y: np.inf)
        np.testing.assert_almost_equal(r[0], 1.0, 8,
                                       "NormInvWish likelihood does not integrate to 1.0")

    # Check that posterior is proportional to prior * likelihood
    D = np.array([[0.1, 0.2], [0.2, 0.3], [0.1, 0.2], [0.4, 0.3]])
    mus = [np.r_[2.1, 1.1], np.r_[0.9, 1.2], np.r_[0.9, 1.1]]
    Sigs = [np.eye(2)*1.5, np.eye(2)*0.7, np.array([[1.1, -0.1], [-0.1, 1.2]])]
    posts = [niw.post(D)(mu, Sig) for mu, Sig in zip(mus, Sigs)]
    posts2 = [niw(mu, Sig)*niw.likelihood(D, mu, Sig) for mu, Sig, in zip(mus, Sigs)]

    np.testing.assert_array_almost_equal(
        posts/posts[0], posts2/posts2[0], 5,
        "NormInvWish posterior not proportional to prior * likelihood.")

    # Check that posterior = prior * likelihood / evidence
    mus = [np.r_[1.1, 1.1], np.r_[1.1, 1.2], np.r_[0.7, 1.3]]
    Sigs = [np.eye(2)*0.2, np.eye(2)*0.1, np.array([[2.1, -0.1], [-0.1, 2.2]])]
    post = niw.post(D)
    post1 = [niw(mu, Sig) * niw.likelihood(D, mu, Sig) / niw.evidence(D)
             for mu, Sig in zip(mus, Sigs)]
    post2 = [post(mu, Sig) for mu, Sig in zip(mus, Sigs)]
    np.testing.assert_array_almost_equal(post1, post2, 10,
                                         "NormInvWish posterior != prior * likelihood / evidence")

   # Would like to check that pred(x) == int prior(theta) * like1(x, theta) d(theta), but I don't
   # know how to integrate over all covariance matrices.  Plus, integrating over a 2D covariance
   # matrix plus a 2D mean is a 5 dimensional integral, which sounds nasty to do.


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--full', action='store_true', help="Run full test suite (slow).")
    args = parser.parse_args()

    test_GaussianMeanKnownVariance()
    test_InvGamma()
    test_InvGamma2D()
    test_NormInvChi2()
    test_NormInvGamma()
    test_NormInvChi2_eq_NormInvGamma()
    test_NormInvWish(args.full)
