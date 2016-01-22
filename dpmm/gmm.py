import numpy as np
from utils import vTmv, pick_discrete


class GaussND(object):
    def __init__(self, mu, Sig):
        self.mu = mu
        self.Sig = Sig
        self.k = len(self.mu)

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

    def __call__(self, x):
        return (np.sqrt((2*np.pi)**self.k*np.linalg.det(self.Sig)) *
                np.exp(-0.5*vTmv(x-self.mu, np.linalg.inv(self.Sig))))

    def sample(self, size=None):
        return np.random.multivariate_normal(mean=self.mu, cov=self.Sig, size=size)


class GMM(object):
    def __init__(self, components, proportions):
        self.components = components
        self.proportions = proportions

    def cond(self, x):
        components = [c.cond(x) for c in self.components]
        return GMM(components, self.proportions)

    def __call__(self, x):
        return np.sum(c(x)*p for c, p in zip(self.components, self.proportions))

    def sample(self, size=None):
        if size is None:
            c = pick_discrete(self.proportions)
            return self.components[c].sample()
        else:
            out = np.empty((size,), dtype=float)
            nums = np.random.multinomial(size, self.proportions)
            i = 0
            for component, num in zip(self.components, nums):
                out[i:i+num] = component.sample(size=num)
                i += num
            return out
