import numpy as np
from utils import pick_discrete


class PseudoMarginalData(object):
    def __init__(self, data, interim_prior):
        # Data should have dims [NOBJ, NSAMPLE, NDIM]
        # interim_prior should have dims [NOBJ, NSAMPLE]
        self.data = data
        self.interim_prior = interim_prior

        self.nobj, self.nsample, self.ndim = self.data.shape

        if self.interim_prior.shape != (self.nobj, self.nsample):
            ds = self.data.shape
            ips = self.interim_prior.shape
            raise ValueError(("data shape [NOBJ, NSAMPLE, NDIM] = [{}, {}, {}]" +
                              " inconsistent with interim_prior shape [NOBJ, NSAMPLE] = [{}, {}]")
                             .format(ds[0], ds[1], ds[2], ips[0], ips[2]))

    def __len__(self):
        return self.nobj

    def __getitem__(self, index):
        import numbers
        cls = type(self)
        # *Leave* a shallow axis in the case a single object is requested.
        if isinstance(index, numbers.Integral):
            return cls(self.data[np.newaxis, index], self.interim_prior[np.newaxis, index])
        else:
            return cls(self.data[index], self.interim_prior[index])

    def random_sample(self):
        """Return a [NOBJ, NDIM] numpy array sampling over NSAMPLE using inverse interim_prior
        weights.  Needed to compute a posterior object."""
        ps = 1./self.interim_prior
        ps /= np.sum(ps, axis=1)[:, np.newaxis]
        return np.array([self.data[i, pick_discrete(p)] for i, p in enumerate(ps)])


class NullManip(object):
    def init(self, D):
        pass

    def __call__(self, D):
        return D

    def unmanip(self, D):
        return D

    def update(self, D, phi, c):
        pass
