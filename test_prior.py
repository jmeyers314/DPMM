import numpy as np
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


def test_GaussianMeanVar():
    mu_0 = 0.0
    kappa_0 = 1
    sigsqr_0 = 1.0
    nu_0 = 1
    model = prior.GaussianMeanVar(mu_0, kappa_0, sigsqr_0, nu_0)

    # Check that we can draw samples from model.
    model.sample()
    model.sample(size=10)

    # Check that we can evaluate a likelihood given 1 data point.
    theta = (1.0, 1.0)
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

if __name__ == "main":
    test_NIW()
    test_GaussianMeanKnownVariance()
    test_GaussianMeanVar()
