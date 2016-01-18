import numpy as np
from collections import namedtuple

SN = namedtuple('SN', ['Mag', 'spec', 'label'])


class SNFamily(object):
    def __init__(self, mean_mag=0.0, std_mag=0.05, mean_spec=0.0, std_spec=1.0, label=None):
        self.mean_mag = mean_mag
        self.std_mag = std_mag
        self.mean_spec = mean_spec
        self.std_spec = std_spec
        self.label = label

    def sample(self, size=None):
        mag = np.random.normal(loc=self.mean_mag, scale=self.std_mag, size=size)
        spec = np.random.normal(loc=self.mean_spec, scale=self.std_spec, size=size)
        if not hasattr(mag, '__len__'):
            return SN(mag, spec, self.label)
        else:
            return [SN(m, s, self.label) for m, s in zip(mag, spec)]


class SNFamilyMixture(object):
    def __init__(self, families, proportions):
        self.families = families
        for i, fam in enumerate(self.families):
            fam.label = i
        self.proportions = proportions

    def sample(self, size=None):
        if size is None:
            ncs = np.random.multinomial(1, self.proportions)
            c = ncs.index(1)
            return self.families[c].sample()
        else:
            out = []
            ncs = np.random.multinomial(size, self.proportions)
            for fam, nc in zip(self.families, ncs):
                out.extend(fam.sample(size=nc))
            return out


def test(size=10000):
    fam1 = SNFamily()
    fam2 = SNFamily(mean_mag=0.1, mean_spec=1.0)
    mixture = SNFamilyMixture([fam1, fam2], [0.3, 0.7])
    return mixture.sample(size=size)
