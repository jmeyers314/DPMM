import numpy as np


class GaussND(object):
    def __init__(self, mu, Sig):
        self.mu = mu
        self.Sig = Sig

    def cond(self, x):
        fixed = np.nonzero([x_ is not None for x_ in x])
        nonfixed = np.nonzero([x_ is None for x_ in x])
        mu1 = self.mu[nonfixed]
        mu2 = self.mu[fixed]
        Sig11 = self.Sig[nonfixed, nonfixed]
        Sig12 = self.Sig[fixed, nonfixed]
        Sig22 = self.Sig[fixed, fixed]

        new_mu = mu1 + np.dot(Sig12, np.dot(np.linalg.inv(Sig22), x[fixed[0]] - mu2))
        new_Sig = Sig11 - np.dot(Sig12, np.dot(np.linalg.inv(Sig22), Sig12.T))
        return GaussND(new_mu, new_Sig)


class GMM(object):
    def __init__(self, components, proportions):
        self.components = components
        self.proportions = proportions

    def cond(self, x):
        components = [c.cond(x) for c in self.components]
        return GMM(components, self.proportions)
