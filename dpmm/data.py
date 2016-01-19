import numpy as np


class PseudoMarginalData(object):
    def __init__(self, data, interim_prior):
        self.data = data
        self.interim_prior = interim_prior

        self.shape = self.data.shape
        self.nobj, self.nsample, self.ndim = self.shape

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
