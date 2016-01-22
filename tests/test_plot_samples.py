import numpy as np
import matplotlib.pyplot as plt
import dpmm
from test_utils import timer


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


if __name__ == "__main__":
    test_GaussianMeanKnownVariance()
    test_InvGamma()
    test_NormInvChi2()
    test_NormInvGamma()
