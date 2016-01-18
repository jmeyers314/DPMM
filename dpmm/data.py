import numpy as np

class PseudoMarginalData(object):
    def __init__(self, data, interim_post):
        self.data = data
        self.interim_post = interim_post

        self.shape = self.data.shape
        self.nobj, self.nsample, self.ndim = self.shape

        if self.interim_post.shape != (self.nobj, self.nsample):
            ds = self.data.shape
            ips = self.interim_post.shape
            raise ValueError(("data shape [NOBJ, NSAMPLE, NDIM] = [{}, {}, {}]" +
                              " inconsistent with interim_post shape [NOBJ, NSAMPLE] = [{}, {}]")
                             .format(ds[0], ds[1], ds[2], ips[0], ips[2]))

    def __getitem__(self, index):
        import numbers
        cls = type(self)
        if isinstance(index, numbers.Integral):
            return cls(self.data[np.newaxis, index], self.interim_post[np.newaxis, index])
        else:
            return cls(self.data[index], self.interim_post[index])
