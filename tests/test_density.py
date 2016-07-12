import warnings
import numpy as np
from scipy.integrate import quad, dblquad

from dpmm.density import t_density, multivariate_t_density, scaled_IX_density
from test_utils import timer


@timer
def test_scaled_IX_density():
    nu = 3
    sigsqr = 1.0

    # test that probability integrates to 1.0
    r = quad(lambda x: scaled_IX_density(nu, sigsqr, x), 0.0, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10, "scaled_IX_density does not integrate to 1.0")

    # test mean
    mean = nu*sigsqr/(nu-2)
    r = quad(lambda x: scaled_IX_density(nu, sigsqr, x)*x, 0.0, np.inf)
    np.testing.assert_almost_equal(r[0], mean, 10, "scaled_IX_density has wrong mean")

    # test variance
    var = 2.0*nu**2*sigsqr/(nu-2.0)**2/(nu-4.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = quad(lambda x: scaled_IX_density(nu, sigsqr, x)*(x-mean)**2, 0.0, np.inf)
    np.testing.assert_almost_equal(r[0], var, 8, "scaled_IX_density has wrong variance")

    # test vectorizability
    x = np.arange(24.0).reshape(4, 3, 2)+1
    prs = scaled_IX_density(nu, sigsqr, x)
    for (i, j, k), pr in np.ndenumerate(prs):
        np.testing.assert_equal(
                pr, scaled_IX_density(nu, sigsqr, x[i, j, k]),
                "scaled_IX_density does not vectorize correctly!")


@timer
def test_t_density():
    nu = 3
    mu = 2.2
    sigsqr = 1.51

    # test that probability integrates to 1.0
    r = quad(lambda x: t_density(nu, mu, sigsqr, x), -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10, "t_density does not integrate to 1.0")

    # test mean
    r = quad(lambda x: t_density(nu, mu, sigsqr, x)*x, -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], mu, 10, "t_density has wrong mean")

    # test variance
    r = quad(lambda x: t_density(nu, mu, sigsqr, x)*(x-mu)**2, -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], nu*sigsqr/(nu-2), 10, "t_density has wrong variance")

    # test vectorizability
    x = np.arange(24.0).reshape(4, 3, 2)+1
    prs = t_density(nu, mu, sigsqr, x)
    for (i, j, k), pr in np.ndenumerate(prs):
        np.testing.assert_equal(
                pr, t_density(nu, mu, sigsqr, x[i, j, k]),
                "t_density does not vectorize correctly!")


@timer
def test_multivariate_t_density(full=False):
    nu = 3
    mu = np.r_[1., 2.]
    Sig = np.eye(2)+0.1

    # test that integrates to 1.0
    r = dblquad(lambda x, y: multivariate_t_density(nu, mu, Sig, np.r_[x, y]),
                -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(
            r[0], 1.0, 5, "multivariate_t_density does not integrate to 1.0")

    if full:
        # test mean
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            xbar = dblquad(lambda x, y: multivariate_t_density(nu, mu, Sig, np.r_[x, y])*x,
                           -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[0]
            ybar = dblquad(lambda x, y: multivariate_t_density(nu, mu, Sig, np.r_[x, y])*y,
                           -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[0]
        np.testing.assert_almost_equal(
                xbar, mu[0], 5, "multivariate_t_density has wrong mean")
        np.testing.assert_almost_equal(
                ybar, mu[1], 5, "multivariate_t_density has wrong mean")
        # test covariance
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Ixx = dblquad(lambda x, y: multivariate_t_density(nu, mu, Sig, np.r_[x, y])*(x-xbar)*(x-xbar),
                          -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[0]
            Iyy = dblquad(lambda x, y: multivariate_t_density(nu, mu, Sig, np.r_[x, y])*(y-ybar)*(y-ybar),
                          -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[0]
            Ixy = dblquad(lambda x, y: multivariate_t_density(nu, mu, Sig, np.r_[x, y])*(x-xbar)*(y-ybar),
                          -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[0]
        cov = np.array([[Ixx, Ixy], [Ixy, Iyy]])
        print cov
        print "----------"
        print nu/(nu-2.)*Sig
        np.testing.assert_almost_equal(
                cov, nu/(nu-2.)*Sig, 2, "multivariate_t_density has wrong covariance")


    # test that we can evaluate multiple probabilities in parallel
    xy1 = np.r_[0.0, 0.1]
    xy2 = np.r_[0.2, 0.3]
    pr1 = [multivariate_t_density(nu, mu, Sig, xy1),
           multivariate_t_density(nu, mu, Sig, xy2)]
    xys = np.vstack([xy1, xy2])
    pr2 = multivariate_t_density(nu, mu, Sig, xys)
    np.testing.assert_array_almost_equal(pr1, pr2, 15, "multivariate_t_density does not vectorize correctly")

    # And a harder, higher dimensional case...
    xys = np.arange(24.0).reshape(4, 3, 2)
    prs = multivariate_t_density(nu, mu, Sig, xys)
    assert prs.shape == (4, 3)
    for (i, j), pr in np.ndenumerate(prs):
        np.testing.assert_array_almost_equal(
                pr, multivariate_t_density(nu, mu, Sig, xys[i, j]), 15,
                "multivariate_t_density does not vectorize correctly")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--full', action='store_true', help="Run full test suite (slow).")
    args = parser.parse_args()

    test_scaled_IX_density()
    test_t_density()
    test_multivariate_t_density(args.full)
