import numpy as np
import matplotlib.pyplot as plt
import dpmm
from test_utils import timer
from dpmm.utils import plot_ellipse, random_wish, random_invwish
from unittest import skip

@skip
@timer
def test_GaussianMeanKnownVariance():
    mu_0 = 1.1
    sigsqr_0 = 0.42
    sigsqr = 0.21
    model = dpmm.GaussianMeanKnownVariance(mu_0, sigsqr_0, sigsqr)

    samples = model.sample(size=1000)

    f = plt.figure(figsize=(5, 3))
    ax = f.add_subplot(111)
    ax.hist(samples, bins=30, normed=True, alpha=0.5, color='k')
    xlim = np.percentile(samples, [1.0, 99.0])
    ax.set_xlim(xlim)
    x = np.linspace(xlim[0], xlim[1], 100)
    y = model(x)
    ax.plot(x, y, c='k', lw=3)
    ax.set_xlabel("$\mu$")
    ax.set_ylabel("Pr($\mu$)")
    f.tight_layout()
    ax.set_title("GaussianMeanKnownVariance")
    f.savefig("plots/GaussianMeanKnownVariance_samples.png")


@skip
@timer
def test_InvGamma():
    alpha = 1.4
    beta = 1.3
    mu = 1.2
    model = dpmm.InvGamma(alpha, beta, mu)

    samples = model.sample(size=1000)
    xlim = np.percentile(samples, [0.0, 95.0])

    f = plt.figure(figsize=(5, 3))
    ax = f.add_subplot(111)
    ax.hist(samples, bins=30, range=xlim, normed=True, alpha=0.5, color='k')
    ax.set_xlim(xlim)
    x = np.linspace(xlim[0], xlim[1], 100)
    y = model(x)
    ax.plot(x, y, c='k', lw=3)
    ax.set_xlabel("$\sigma^2$")
    ax.set_ylabel("Pr($\sigma^2$)")
    f.tight_layout()
    ax.set_title("InvGamma")
    f.savefig("plots/InvGamma_samples.png")


@skip
@timer
def test_NormInvChi2():
    mu_0 = 1.5
    kappa_0 = 2.3
    sigsqr_0 = 0.24
    nu_0 = 2
    model = dpmm.NormInvChi2(mu_0, kappa_0, sigsqr_0, nu_0)

    samples = model.sample(size=1000)
    mu_samples = np.array([s[0] for s in samples])
    var_samples = np.array([s[1] for s in samples])

    xlim = np.percentile(mu_samples, [2.5, 97.5])
    f = plt.figure(figsize=(5, 3))
    ax = f.add_subplot(111)
    ax.hist(mu_samples, bins=30, range=xlim, normed=True, alpha=0.5, color='k')
    ax.set_xlim(xlim)
    x = np.linspace(xlim[0], xlim[1], 100)
    y = model.marginal_mu(x)
    ax.plot(x, y, c='k', lw=3)
    ax.set_xlabel("$\mu$")
    ax.set_ylabel("Pr($\mu$)")
    f.tight_layout()
    ax.set_title("NormInvChi2")
    f.savefig("plots/NormInvChi2_mu_samples.png")

    xlim = np.percentile(var_samples, [0.0, 95.0])
    f = plt.figure(figsize=(5, 3))
    ax = f.add_subplot(111)
    ax.hist(var_samples, bins=30, range=xlim, normed=True, alpha=0.5, color='k')
    ax.set_xlim(xlim)
    x = np.linspace(xlim[0], xlim[1], 100)
    y = model.marginal_var(x)
    ax.plot(x, y, c='k', lw=3)
    ax.set_xlabel("$\sigma^2$")
    ax.set_ylabel("Pr($\sigma^2$)")
    f.tight_layout()
    ax.set_title("NormInvChi2")
    f.savefig("plots/NormInvChi2_var_samples.png")

@skip
@timer
def test_NormInvGamma():
    mu_0 = 1.5
    V_0 = 1.2
    a_0 = 1.24
    b_0 = 1.1
    model = dpmm.NormInvGamma(mu_0, V_0, a_0, b_0)

    samples = model.sample(size=1000)
    mu_samples = np.array([s[0] for s in samples])
    var_samples = np.array([s[1] for s in samples])

    xlim = np.percentile(mu_samples, [2.5, 97.5])
    f = plt.figure(figsize=(5, 3))
    ax = f.add_subplot(111)
    ax.hist(mu_samples, bins=30, range=xlim, normed=True, alpha=0.5, color='k')
    ax.set_xlim(xlim)
    x = np.linspace(xlim[0], xlim[1], 100)
    y = model.marginal_mu(x)
    ax.plot(x, y, c='k', lw=3)
    ax.set_xlabel("$\mu$")
    ax.set_ylabel("Pr($\mu$)")
    f.tight_layout()
    ax.set_title("NormInvGamma")
    f.savefig("plots/NormInvGamma_mu_samples.png")

    xlim = np.percentile(var_samples, [0.0, 95.0])
    f = plt.figure(figsize=(5, 3))
    ax = f.add_subplot(111)
    ax.hist(var_samples, bins=30, range=xlim, normed=True, alpha=0.5, color='k')
    ax.set_xlim(xlim)
    x = np.linspace(xlim[0], xlim[1], 100)
    y = model.marginal_var(x)
    ax.plot(x, y, c='k', lw=3)
    ax.set_xlabel("$\sigma^2$")
    ax.set_ylabel("Pr($\sigma^2$)")
    f.tight_layout()
    ax.set_title("NormInvGamma")
    f.savefig("plots/NormInvGamma_var_samples.png")


@skip
@timer
def test_NormInvWish():
    mu_0 = np.r_[0.3, -0.2]
    d = len(mu_0)
    Lam_0 = np.linalg.inv(np.array([[2, 1.1], [1.1, 1.2]]))
    kappa_0 = 2.1
    nu_0 = 8

    model = dpmm.NormInvWish(mu_0, kappa_0, Lam_0, nu_0)

    # First check some numerics
    Nsample = 5000
    samples = model.sample(size=Nsample)
    mu_samples = [s[0] for s in samples]
    cov_samples = [s[1] for s in samples]

    mean = np.mean(mu_samples, axis=0)
    std = np.std(mu_samples, axis=0)/np.sqrt(Nsample)
    print "NormInvWish mu_0 = {}".format(mu_0)
    print "NormInvWish E(mu) = {} +/- {}".format(mean, std)

    mean_cov = np.mean(cov_samples, axis=0)
    std_cov = np.std(cov_samples, axis=0)/np.sqrt(Nsample)
    print "NormInvWish (Lam_0)^(-1)/(nu_0-d-1) = \n{}".format(np.linalg.inv(Lam_0)/(nu_0-d-1))
    print "NormInvWish E(Sig) = \n{}\n +/-\n{}".format(mean_cov, std_cov)

    # Now try some plots with different values of kappa_0 and nu_0
    f = plt.figure(figsize=(7, 7))
    for i, (kappa_0, nu_0) in enumerate(zip([0.4, 0.4, 6.5, 6.5],
                                            [10, 4, 10, 4])):
        model = dpmm.NormInvWish(mu_0, kappa_0, Lam_0, nu_0)
        samples = model.sample(size=25)
        ax = f.add_subplot(2, 2, i+1)
        for sample in samples:
            mu, Sig = sample
            plot_ellipse(mu, Sig, ax=ax, facecolor='none', edgecolor='k', alpha=0.2)
        plot_ellipse(mu_0, np.linalg.inv(Lam_0)/(nu_0-d-1), ax=ax, facecolor='none', edgecolor='r')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axvline(mu_0[0], c='r', alpha=0.1)
        ax.axhline(mu_0[1], c='r', alpha=0.1)
        ax.set_title(r"$\kappa_0$={}, $\nu_0$={}".format(kappa_0, nu_0))
        print np.mean([s[1] for s in samples], axis=0)
    f.savefig("plots/NormInvWish_samples.png")


@skip
@timer
def test_random_wish():
    dof = 3
    S = np.array([[1.0, 0.25], [0.25, 0.5]])
    Nsamples = 5000
    samples = random_wish(dof, S, size=Nsamples)
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)/np.sqrt(Nsamples)

    print "E(wish) = \n{}".format(dof * S)
    print "<wish> = \n{}\n +/-\n{}".format(mean, std)


@skip
@timer
def test_random_invwish():
    dof = 6
    d = 2
    S = np.array([[1.0, 0.25], [0.25, 0.5]])
    invS = np.linalg.inv(S)
    Nsamples = 5000
    samples = random_invwish(dof, invS, size=Nsamples)
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)/np.sqrt(Nsamples)

    print "E(invwish) = \n{}".format(S/(dof-d-1))
    print "<invwish> = \n{}\n +/-\n{}".format(mean, std)


@skip
@timer
def test_ellipse_plotter():
    f = plt.figure(figsize=(7, 7))
    for i, Sig in enumerate([np.array([[1.0, 0.0], [0.0, 0.25]]),
                             np.array([[0.25, 0.0], [0.0, 1.0]]),
                             np.array([[1.0, 0.8], [0.8, 1.0]]),
                             np.array([[1.0, -0.8], [-0.8, 1.0]])]):
        ax = f.add_subplot(2, 2, i+1)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        plot_ellipse([0., 0.], Sig)
        ax.set_title("$\Sigma$={}".format(Sig))
    f.tight_layout()
    f.savefig("plots/ellipse.png")

if __name__ == "__main__":
    test_GaussianMeanKnownVariance()
    test_InvGamma()
    test_NormInvChi2()
    test_NormInvGamma()
    test_NormInvWish()
    test_random_wish()
    test_random_invwish()
    test_ellipse_plotter()
