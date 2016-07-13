import numpy as np


class GaussND(object):
    def __init__(self, mu, Sig):
        self.mu = np.atleast_1d(mu)
        self.Sig = np.atleast_2d(Sig)
        self.d = len(self.mu)

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

    def sample(self, size=None):
        if self.d == 1:
            return np.random.normal(self.mu, scale=np.sqrt(self.Sig), size=size)
        else:
            return np.random.multivariate_normal(self.mu, self.Sig, size=size)


class GMM(object):
    def __init__(self, components, proportions):
        self.components = components
        self.proportions = proportions
        self.d = self.components[0].d

    def cond(self, x):
        components = [c.cond(x) for c in self.components]
        return GMM(components, self.proportions)

    def sample(self, size=None):
        if size is None:
            nums = np.random.multinomial(1, self.proportions)
            c = nums.index(1) # which class got picked
            return self.components[c].sample()
        else:
            n = np.prod(size)
            if self.d == 1:
                out = np.empty((n,), dtype=float)
                nums = np.random.multinomial(n, self.proportions)
                i = 0
                for component, num in zip(self.components, nums):
                    out[i:i+num] = component.sample(size=num)
                    i += num
                out = out.reshape(size)
            else:
                out = np.empty((n, self.d), dtype=float)
                nums = np.random.multinomial(n, self.proportions)
                i = 0
                for component, num in zip(self.components, nums):
                    out[i:i+num] = component.sample(size=num)
                    i += num
                if isinstance(size, int):
                    out = out.reshape((size, self.d))
                else:
                    out = out.reshape(size+(self.d,))
            return out
