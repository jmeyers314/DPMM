import numpy as np
from scipy.special import gamma
import bisect


def vTmv(vec, mat=None, vec2=None):
    """Multiply a vector transpose times a matrix times a vector.

    @param vec  The first vector (will be transposed).
    @param mat  The matrix in the middle.  Identity by default.
    @param vec2 The second vector (will not be transposed.)  By default, the same as the vec.
    @returns    Product.  Could be a scalar or a matrix depending on whether vec is a row or column
                vector.
    """
    if len(vec.shape) == 1:
        vec = np.reshape(vec, [vec.shape[0], 1])
    if mat is None:
        mat = np.eye(len(vec))
    if vec2 is None:
        vec2 = vec
    return np.dot(vec.T, np.dot(mat, vec2))


def gammad(d, nu_over_2):
    """D-dimensional gamma function."""
    nu = 2.0 * nu_over_2
    return np.pi**(d*(d-1.)/4)*np.multiply.reduce([gamma(0.5*(nu+1-i)) for i in range(d)])


def random_wish(dof, S, size=1):
    dim = S.shape[0]
    if size == 1:
        x = np.random.multivariate_normal(np.zeros(dim), S, size=dof)
        return np.dot(x.T, x)
    else:
        out = np.empty((size, dim, dim), dtype=np.float64)
        for i in range(size):
            x = np.random.multivariate_normal(np.zeros(dim), S, size=dof)
            out[i] = np.dot(x.T, x)
        return out


def random_invwish(dof, invS, size=1):
    return np.linalg.inv(random_wish(dof, invS, size=size))


def pick_discrete(p):
    """Pick a discrete integer between 0 and len(p) - 1 with probability given by p array."""
    c = np.cumsum(p)
    u = np.random.uniform()
    return bisect.bisect_left(c, u)
