import itertools

import numpy as np
from utils import pick_discrete
from data import PseudoMarginalData, NullManip


class DPMM(object):
    """Dirichlet Process Mixture Model.  Using algorithm 2 from Neal (2000).

    @param prior   The prior object for whatever model is being inferred.
    @param alpha   DP concentration parameter.
    @param D       Data.
    @param manip   A data manipulator.  Used for unshearing, for example.
    @param phi     Optional initial state for each cluster.
    @param label   Optional initial cluster labels for each data point.
    """
    def __init__(self, prior, alpha, D, manip=None, phi=None, label=None):
        self.prior = prior
        self.alpha = alpha
        self._D = D  # data
        if manip is None:
            manip = NullManip()
        self.manip = manip

        self._initD()
        self.manip.init(self.D)

        self.n = len(self.D)

        # Initialize r_i array
        # This is Neal (2000) equation (3.4) without the b factor.
        self.r_i = self.alpha * self.prior.pred(self.mD)

        if phi is None:
            self.init_phi()
        else:
            self.phi = phi
            self.label = label
            self.nphi = [np.sum(label == i) for i in xrange(label.max())]

    def init_phi(self):
        self.label = np.zeros((self.n), dtype=int)
        self.phi = []
        self.nphi = []
        for i in xrange(self.n):
            self.update_c_i(i)

    @property
    def mD(self):
        if self.manip_needs_update:
            self._mD = self.manip(self.D)
            self.manip_needs_update = False
        return self._mD

    def _initD(self):
        """Initialize latent data vector."""
        if isinstance(self._D, PseudoMarginalData):
            self.D = np.mean(self._D.data, axis=1)
        else:
            self.D = self._D
        self.manip_needs_update = True

    def draw_new_label(self, i):
        # This is essentially Neal (2000) equation (3.6)
        # Start off with the probabilities for cloning an existing cluster:
        p = [l1 * nphi
             for l1, nphi in itertools.izip(self.prior.like1N(self.mD[i], self.phi),
                                            self.nphi)]
        # and then append the probability to create a new cluster.
        p.append(self.r_i[i])
        p = np.array(p)
        # Normalize.  This essentially takes care of the factors of b/(n-1+alpha) in Neal (2000)
        # equation (3.6)
        p /= np.sum(p)
        picked = pick_discrete(p)
        return picked

    def del_c_i(self, i):
        # De-associate the ith data point from its cluster.
        label = self.label[i]
        # We're about to assign this point to a new cluster, so decrement current cluster count.
        self.nphi[label] -= 1
        # If we just deleted the last cluster member, then delete the cluster from self.phi
        if self.nphi[label] == 0:
            del self.phi[label]
            del self.nphi[label]
            # Need to decrement label numbers for labels greater than the one deleted...
            self.label[np.nonzero(self.label >= label)] -= 1

    def update_c_i(self, i):
        # for deduplication
        # Neal (2000) equation 3.6.  See draw_new_label above.
        new_label = self.draw_new_label(i)
        self.label[i] = new_label
        # If we selected to create a new cluster, then draw parameters for that cluster.
        if new_label == len(self.phi):
            self.phi.append(self.prior.post(self.mD[i]).sample())
            self.nphi.append(1)
        else:  # Otherwise just increment the count for the cloned cluster.
            self.nphi[new_label] += 1

    def update_c(self):
        # This is the first bullet for Neal (2000) algorithm 2, updating the labels for each data
        # point and potentially deleting clusters that are no longer populated or creating new
        # clusters with probability proportional to self.alpha.
        for i in xrange(self.n):
            self.del_c_i(i)
            self.update_c_i(i)

    def update_phi(self):
        # This is the second bullet for Neal (2000) algorithm 2, updating the parameters phi of each
        # cluster conditional on that clusters currently associated data members.
        for i in xrange(len(self.phi)):
            data = self.mD[np.nonzero(self.label == i)]
            self.phi[i] = self.prior.post(data).sample()

    def update_latent_data(self):
        # Update the latent "true" data in the case that the data is represented by a
        # Pseudo-marginal samples or (TBD) means and Gaussian errors.
        if isinstance(self._D, PseudoMarginalData):
            for i, ph in enumerate(self.phi):
                index = np.nonzero(self.label == i)
                data = self._D[index]  # a PseudoMarginalData instance
                # calculate weights for selecting a representative sample
                ps = self.prior.like1(self.manip(data.data), *ph)[..., 0] / data.interim_prior
                ps /= np.sum(ps, axis=1)[:, np.newaxis]
                for j, p in enumerate(ps):
                    self.D[index[0][j]] = data.data[j, pick_discrete(p)]
            # Need to update the r_i probabilities too since self.D changed.
            self.r_i = self.alpha * self.prior.pred(self.mD)
            self.manip_needs_update = True
        else:
            pass  # If data is already a numpy array, there's nothing to update.

    def update(self, n=1):
        # Neal (2000) algorithm 2.
        for j in xrange(n):
            self.update_c()
            self.update_latent_data()
            self.update_phi()
            # Give manip.update() the *unmanipulated* data.
            self.manip.update(self.D, self.phi, self.label, self.prior)
            self.manip_needs_update = True
