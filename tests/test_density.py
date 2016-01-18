import warnings
import numpy as np
from scipy.integrate import quad, dblquad

import dpmm
from test_utils import timer


@timer
def test_scaled_IX_density():
    nu = 3
    sigsqr = 1.0
    r = quad(lambda x: dpmm.density.scaled_IX_density(nu, sigsqr, x), 0.0, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10, "scaled_IX_density does not integrate to 1.0")
    # test mean
    mean = nu*sigsqr/(nu-2)
    r = quad(lambda x: dpmm.density.scaled_IX_density(nu, sigsqr, x)*x, 0.0, np.inf)
    np.testing.assert_almost_equal(r[0], mean, 10, "scaled_IX_density has wrong mean")
    # test variance
    var = 2.0*nu**2*sigsqr/(nu-2.0)**2/(nu-4.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = quad(lambda x: dpmm.density.scaled_IX_density(nu, sigsqr, x)*(x-mean)**2, 0.0, np.inf)
    np.testing.assert_almost_equal(r[0], var, 8, "scaled_IX_density has wrong variance")


@timer
def test_t_density():
    nu = 3
    mu = 2.2
    sigsqr = 1.51
    r = quad(lambda x: dpmm.density.t_density(nu, mu, sigsqr, x), -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 10, "t_density does not integrate to 1.0")
    # test mean
    r = quad(lambda x: dpmm.density.t_density(nu, mu, sigsqr, x)*x, -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], mu, 10, "t_density has wrong mean")
    # test variance
    r = quad(lambda x: dpmm.density.t_density(nu, mu, sigsqr, x)*(x-mu)**2, -np.inf, np.inf)
    np.testing.assert_almost_equal(r[0], nu*sigsqr/(nu-2), 10, "t_density has wrong variance")


@timer
def test_multivariate_t_density():
    nu = 3
    mu = np.r_[1., 2.]
    Sig = np.eye(2)+0.1

    # test that integrates to 1.0
    r = dblquad(lambda x, y: dpmm.density.multivariate_t_density(nu, mu, Sig, [x, y]),
                -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(r[0], 1.0, 5,
                                   "multivariate_t_density does not integrate to 1.0")

    # test mean
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xbar = dblquad(lambda x, y: dpmm.density.multivariate_t_density(nu, mu, Sig, [x, y])*x,
                       -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
        ybar = dblquad(lambda x, y: dpmm.density.multivariate_t_density(nu, mu, Sig, [x, y])*y,
                       -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
    np.testing.assert_almost_equal(xbar[0], mu[0], 5,
                                   "multivariate_t_density has wrong mean")
    np.testing.assert_almost_equal(ybar[0], mu[1], 5,
                                   "multivariate_t_density has wrong mean")

    # test that we can evaluate multiple probabilities in parallel
    xy1 = np.r_[0.0, 0.1]
    xy2 = np.r_[0.2, 0.3]
    pr1 = [dpmm.density.multivariate_t_density(nu, mu, Sig, xy1),
           dpmm.density.multivariate_t_density(nu, mu, Sig, xy2)]
    xys = np.vstack([xy1, xy2])
    pr2 = dpmm.density.multivariate_t_density(nu, mu, Sig, xys)
    np.testing.assert_array_almost_equal(pr1, pr2, 10,
                                         "multivariate_t_density does not parallelize correctly")


if __name__ == "__main__":
    test_t_density()
    test_scaled_IX_density()
    test_multivariate_t_density()
