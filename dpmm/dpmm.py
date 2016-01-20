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
            # A few points:
            # 1) It's easier to create new clusters than to destroy existing clusters, so start off
            #    with all data points in a single cluster.
            # 2) Store cluster params in a list so that it's easy to expand/contract as the sampler
            #    samples different numbers of components.
            # 3) Finally, each element of phi should be a tuple so that we can call
            #    prior.like1(x, *phi).
            phi = [self.prior.post(D).sample()]
            nphi = [self.n]  # Number of data points assigned to each tuple.
            label = np.zeros((self.n), dtype=int)  # cluster assignment for each data point.
        if nphi is None:
            nphi = [np.sum(label == i) for i in xrange(label.max())]

        self.phi = phi
        self.nphi = nphi
        self.label = label

        # Initialize r_i array
        # This is Neal (2000) equation (3.4) without the b factor.
        self.r_i = self.alpha * prior.pred(D)

    def draw_new_label(self, i):
        # This is essentially Neal (2000) equation (3.6)
        # Start off with the probabilities for cloning an existing cluster:
        p = [self.prior.like1(self.D[i], *phi)*nphi
             for phi, nphi in itertools.izip(self.phi, self.nphi)]
        # and then append the probability to create a new cluster.
        p.append(self.r_i[i])
        p = np.array(p)
        # Normalize.  This essentially takes care of the factors of b/(n-1+alpha) in Neal (2000)
        # equation (3.6)
        p /= np.sum(p)
        picked = pick_discrete(p)
        return picked

    def update_c(self):
        # This is the first bullet for Neal (2000) algorithm 2, updating the labels for each data
        # point and potentially deleting clusters that are no longer populated or creating new
        # clusters with probability proportional to self.alpha.
        for i in xrange(self.n):
            label = self.label[i]
            # We're about to assign this point to a new cluster, so decrement current cluster count.
            self.nphi[label] -= 1
            # If we just deleted the last cluster member, then delete the cluster from self.phi
            if self.nphi[label] == 0:
                del self.phi[label]
                del self.nphi[label]
                # Need to decrement label numbers for labels greater than the one deleted...
                self.label[np.nonzero(self.label >= label)] -= 1
            # Neal (2000) equation 3.6.  See function above.
            new_label = self.draw_new_label(i)
            self.label[i] = new_label
            # If we selected to create a new cluster, then draw parameters for that cluster.
            if new_label == len(self.phi):
                self.phi.append(self.prior.post(self.D[i]).sample())
                self.nphi.append(1)
            else:  # Otherwise just increment the count for the cloned cluster.
                self.nphi[new_label] += 1

    def update_phi(self):
        # This is the second bullet for Neal (2000) algorithm 2, updating the parameters phi of each
        # cluster conditional on that clusters currently associated data members.
        for i in xrange(len(self.phi)):
            data = self.D[np.nonzero(self.label == i)]
            self.phi[i] = self.prior.post(data).sample()

    def update(self, n=1):
        # Neal (2000) algorithm 2.
        for j in xrange(n):
            self.update_c()
            self.update_phi()
