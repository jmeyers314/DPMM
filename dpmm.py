""" Going to try algorithm 1 from Neal (2000)
"""

import numpy as np
import bisect


def pick_discrete(p):
    """Pick a discrete integer between 0 and len(p) - 1 with probability given by p array."""
    c = np.cumsum(p)
    u = np.random.uniform()
    return bisect.bisect_left(c, u)


class DPMM(object):
    """Dirichlet Process Mixture Model.  Using algorithm 1 from Neal (2000).

    @param conjugate_prior  The conjugate_prior object for whatever model is being inferred.
    @param alpha            Concentration parameter.
    @param D                Data.
    @param theta            Optional initial state for sampler.  Will be drawn from conjugate_prior
                            if not specified.
    """
    def __init__(self, conjugate_prior, alpha, D, theta=None):
        self.conjugate_prior = conjugate_prior
        self.alpha = alpha
        self.D = D  # data
        self.n = len(self.D)

        if theta is None:
            # Draw from the prior
            theta = [tuple(conjugate_prior.sample()) for i in range(self.n)]
        self.theta = theta

        # Initialize r_i array
        self.r_i = self.alpha * np.array([conjugate_prior.pred(x) for x in D])

    def q(self, i):
        # compute and return row of q_ij matrix (we only ever need one row at a time).
        qs = np.array([self.conjugate_prior.like1(*th_j, x=self.D[i]) for th_j in self.theta])
        qs[i] = self.r_i[i]  # cheat by placing r_i at q_ii.
        return qs

    def update_1_theta(self, i):
        x = self.D[i]
        qs = self.q(i)
        p = qs/np.sum(qs)
        picked = pick_discrete(p)
        if picked == i:  # This corresponds to picking r_i in Neal (2000); i.e. get a new theta
            # Neal (2000) H_i is the posterior given a single observation x.
            self.theta[i] = tuple(self.conjugate_prior.post(x).sample())
        else:  # reuse an existing theta
            self.theta[i] = self.theta[picked]

    def update(self, n=1):
        for j in xrange(n):
            for i in xrange(len(self.D)):
                self.update_1_theta(i)


class DPMM2(object):
    """Dirichlet Process Mixture Model.  Using algorithm 2 from Neal (2000).

    @param conjugate_prior  The conjugate_prior object for whatever model is being inferred.
    @param alpha            Concentration parameter.
    @param D                Data.
    @param phi              Optional initial state for sampler.
    @param label            Optional initial class labels for sampler.
    """
    def __init__(self, conjugate_prior, alpha, D, phi=None, label=None):
        self.conjugate_prior = conjugate_prior
        self.alpha = alpha
        self.D = D  # data
        self.n = len(self.D)

        if phi is None:
            # Draw from the prior.  Use a list for this one, since the length will be changing.
            phi = [tuple(self.conjugate_prior.sample()) for i in xrange(self.n)]
            nphi = [1]*self.n
            # Give each sample it's own class label.
            label = np.arange(self.n)

        self.phi = phi
        self.nphi = nphi
        self.label = label

        # Initialize r_i array again
        self.r_i = self.alpha * np.array([conjugate_prior.pred(x) for x in D])

    def draw_new_label(self, i):
        p = [self.conjugate_prior.like1(*phi, x=self.D[i])*nphi
             for phi, nphi in zip(self.phi, self.nphi)]
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
                self.phi.append(tuple(self.conjugate_prior.post(self.D[i]).sample()))
                self.nphi.append(1)
            else:
                self.nphi[new_label] += 1

    def update_phi(self):
        for i in xrange(len(self.phi)):
            data = self.D[np.nonzero(self.label == i)]
            self.phi[i] = tuple(self.conjugate_prior.post(data).sample())

    def update(self, n=1):
        for j in xrange(n):
            self.update_c()
            self.update_phi()
