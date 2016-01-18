""" Going to try algorithm 1 from Neal (2000)
"""

import itertools

import numpy as np
from utils import pick_discrete


class DPMM(object):
    """Dirichlet Process Mixture Model.  Using algorithm 2 from Neal (2000).

    @param prior    The prior object for whatever model is being inferred.
    @param alpha    Concentration parameter.
    @param D        Data.
    @param phi      Optional initial state for each cluster.
    @param label    Optional initial cluster labels for each data point.
    """
    def __init__(self, prior, alpha, D, phi=None, label=None):
        self.prior = prior
        self.alpha = alpha
        self.D = D  # data
        self.n = len(self.D)

        if phi is None:
            # It's easier to create new clusters than to destroy existing clusters, so start off
            # with all data points in a single cluster.
            phi = [self.prior.post(D).sample()]
            nphi = [self.n]
            label = np.zeros((self.n), dtype=int)
        if nphi is None:
            nphi = [np.sum(label == i) for i in xrange(label.max())]

        self.phi = phi
        self.nphi = nphi
        self.label = label

        # Initialize r_i array again
        self.r_i = self.alpha * prior.pred(D)

    def draw_new_label(self, i):
        p = [self.prior.like1(*phi, x=self.D[i])*nphi
             for phi, nphi in itertools.izip(self.phi, self.nphi)]
        p.append(self.r_i[i])
        p = np.array(p)
        p /= np.sum(p)
        picked = pick_discrete(p)
        return picked

    def update_c(self):
        for i in xrange(self.n):
            label = self.label[i]
            self.nphi[label] -= 1
            if self.nphi[label] == 0:
                del self.phi[label]
                del self.nphi[label]
                # Need to decrement labels for label beyond the one deleted...
                self.label[np.nonzero(self.label >= label)] -= 1
            new_label = self.draw_new_label(i)
            self.label[i] = new_label
            if new_label == len(self.phi):
                self.phi.append(self.prior.post(self.D[i]).sample())
                self.nphi.append(1)
            else:
                self.nphi[new_label] += 1

    def update_phi(self):
        for i in xrange(len(self.phi)):
            data = self.D[np.nonzero(self.label == i)]
            self.phi[i] = self.prior.post(data).sample()

    def update(self, n=1):
        for j in xrange(n):
            self.update_c()
            self.update_phi()
