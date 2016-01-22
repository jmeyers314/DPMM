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


# Modified code from http://stackoverflow.com/questions/9081553/python-scatter-plot-size-and-style-of-the-marker/24567352#24567352

def ellipses(x, y, s, q, pa, c='b', ax=None, vmin=None, vmax=None, **kwargs):
    """Scatter plot of ellipses.

    (x, y) duh.
    s      size.
    q      minor-to-major axes ratio b/a
    pa     position angle in deg, CCW from +y.
    """
    from matplotlib.patches import Ellipse
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if isinstance(c, basestring):
        color = c     # ie. use colors.colorConverter.to_rgba_array(c)
    else:
        color = None  # use cmap, norm after collection is created
    kwargs.update(color=color)

    w, h = s*np.sqrt(q), s/np.sqrt(q)

    if np.isscalar(x):
        patches = [Ellipse((x, y), w, h, pa), ]
    else:
        patches = [Ellipse((x_, y_), w_, h_, pa_) for x_, y_, w_, h_, pa_ in zip(x, y, w, h, pa)]
    collection = PatchCollection(patches, **kwargs)

    if color is None:
        collection.set_array(np.asarray(c))
        if vmin is not None or vmax is not None:
            collection.set_clim(vmin, vmax)

    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def plot_ellipse(mu, Sig, ax=None, **kwargs):
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    val, vec = np.linalg.eigh(Sig)
    # 5.991 gives 95% ellipses
    s = np.sqrt(np.sqrt(5.991*val[0]*val[1]))
    q = np.sqrt(val[0]/val[1])
    pa = np.arctan2(vec[0, 1], vec[0, 0])*180/np.pi
    ellipses(mu[0], mu[1], s, q, pa, ax=ax, **kwargs)
