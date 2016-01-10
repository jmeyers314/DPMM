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
    """Dirichlet Process Mixture Model.

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
            theta = conjugate_prior.sample(size=len(D))
        self.theta = theta

        # Initialize r_i array
        self.r_i = self.alpha * np.array([conjugate_prior.pred(x) for x in D])

    def q(self, i):
        # compute and return row of q_ij matrix (we only ever need one row at a time).
        qs = np.array([self.conjugate_prior.like1(th_j, x=self.D[i]) for th_j in self.theta])
        qs[i] = self.r_i[i]  # cheat by placing r_i at q_ii.
        return qs

    def update_1_theta(self, i):
        x = self.D[i]
        qs = self.q(i)
        p = qs/np.sum(qs)
        picked = pick_discrete(p)
        if picked == i:  # This corresponds to picking r_i in Neal (2000); i.e. get a new theta
            # Neal (2000) H_i is the posterior given a single observation x.
            self.theta[i] = self.conjugate_prior.post(x).sample()
        else:  # reuse an existing theta
            self.theta[i] = self.theta[picked]

    def update_theta(self, n=1):
        for j in xrange(n):
            for i in xrange(len(self.D)):
                self.update_1_theta(i)
