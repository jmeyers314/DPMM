import numpy as np
import dpmm
from test_utils import timer

@timer
def test_InvGamma():
    """Test broadcasting rules for InvGamma prior."""

    # Test sample() method:
    prior = dpmm.InvGamma(1.0, 1.0, 0.0)
    arr = prior.sample()
    assert isinstance(arr, float)

    arr = prior.sample(size=1)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1,)
    assert arr.dtype == float

    arr = prior.sample(size=(1,))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1,)
    assert arr.dtype == float

    arr = prior.sample(size=10)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (10,)
    assert arr.dtype == float

    arr = prior.sample(size=(10, 20))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (10, 20)
    assert arr.dtype == float

    # Test like1() method:
    prior = dpmm.InvGamma(1.0, 1.0, 0.0)
    x = 1.0
    var = 1.0
    arr = prior.like1(x, var)
    assert isinstance(arr, float)

    x = np.array([1.0])
    arr = prior.like1(x, var)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1,)
    assert arr.dtype == float

    x = np.array([1.0, 2.0])
    arr = prior.like1(x, var)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2,)
    assert arr.dtype == float

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    arr = prior.like1(x, var)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)
    assert arr.dtype == float

    x = np.array([1.0, 2.0])
    var = np.array([2.0, 3.0])
    arr = prior.like1(x, var)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2,)
    assert arr.dtype == float

    x = np.array([1.0, 2.0])
    var = np.array([1.0, 2.0, 3.0])
    arr = prior.like1(x[:, np.newaxis], var)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)
    assert arr.dtype == float
    arr = prior.like1(x, var[:, np.newaxis])
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 2)
    assert arr.dtype == float

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = np.array([10.0, 11.0, 12.0, 13.0])
    arr = prior.like1(x[:, :, np.newaxis], y)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3, 4)
    assert arr.dtype == float
    arr = prior.like1(x, y[:, np.newaxis, np.newaxis])
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4, 2, 3)
    assert arr.dtype == float

    # Test __call__() method:
    prior = dpmm.InvGamma(1.0, 1.0, 0.0)
    var = 1.0
    arr = prior(var)
    assert isinstance(arr, float)

    var = np.array([1.0])
    arr = prior(var)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1,)
    assert arr.dtype == float

    var = np.array([1.0, 2.0])
    arr = prior(var)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2,)
    assert arr.dtype == float

    var = np.array([[1.0, 2.0], [3.0, 4.0]])
    arr = prior(var)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)
    assert arr.dtype == float

    # Should _post_params method do any broadcasting?

    # Test pred method():
    prior = dpmm.InvGamma(1.0, 1.0, 0.0)
    x = 1.0
    arr = prior.pred(x)
    assert isinstance(arr, float)

    x = np.array([1.0])
    arr = prior.pred(x)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1,)
    assert arr.dtype == float

    x = np.array([1.0, 2.0])
    arr = prior.pred(x)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2,)
    assert arr.dtype == float

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    arr = prior.pred(x)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)
    assert arr.dtype == float


@timer
def test_NormInvGamma():
    """Test broadcasting rules for NormInvGamma prior."""

    # Test sample() method:
    prior = dpmm.NormInvGamma(1.0, 1.0, 1.0, 1.0)
    arr = prior.sample()
    assert isinstance(arr, np.void)
    assert arr.dtype == prior.model_dtype

    arr = prior.sample(size=1)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1,)
    assert arr.dtype == prior.model_dtype

    arr = prior.sample(size=(1,))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1,)
    assert arr.dtype == prior.model_dtype

    arr = prior.sample(size=10)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (10,)
    assert arr.dtype == prior.model_dtype

    arr = prior.sample(size=(10, 20))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (10, 20)
    assert arr.dtype == prior.model_dtype

    # Test like1() method:
    prior = dpmm.NormInvGamma(1.0, 1.0, 1.0, 1.0)
    x = 1.0
    mu = 1.0
    var = 1.0
    arr = prior.like1(x, mu, var)
    assert isinstance(arr, float)

    x = np.array([1.0])
    mu = 1.0
    var = 1.0
    arr = prior.like1(x, mu, var)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1,)
    assert arr.dtype == float

    x = np.array([1.0, 2.0, 3.0, 4.0])
    mu = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    arr = prior.like1(x[:, np.newaxis, np.newaxis], mu, var)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4, 2, 3)
    assert arr.dtype == float

    theta = np.zeros((2, 3), dtype=prior.model_dtype)
    theta['mu'] = mu
    theta['var'] = var
    arr = prior.like1(x[:, np.newaxis, np.newaxis], theta)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4, 2, 3)
    assert arr.dtype == float

    arr = prior.like1(x, mu[:, :, np.newaxis], var[:, :, np.newaxis])
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3, 4)
    assert arr.dtype == float

    arr = prior.like1(x, theta[:, :, np.newaxis])
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3, 4)
    assert arr.dtype == float

    # Test __call__() method:
    prior = dpmm.NormInvGamma(1.0, 1.0, 1.0, 1.0)
    mu = 1.0
    var = 1.0
    arr = prior(mu, var)
    assert isinstance(arr, float)

    mu = np.array([1.0])
    arr = prior(mu, var)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1,)
    assert arr.dtype == float

    mu = np.array([1.0, 2.0])
    var = np.array([10.0, 11.0, 12.0])
    arr = prior(mu[:, np.newaxis], var)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)
    assert arr.dtype == float

    # Should _post_params method do any broadcasting?

    # Test pred method():
    prior = dpmm.NormInvGamma(1.0, 1.0, 1.0, 1.0)
    x = 1.0
    arr = prior.pred(x)
    assert isinstance(arr, float)

    x = np.array([1.0])
    arr = prior.pred(x)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1,)
    assert arr.dtype == float

    x = np.array([1.0, 2.0])
    arr = prior.pred(x)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2,)
    assert arr.dtype == float

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    arr = prior.pred(x)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)
    assert arr.dtype == float


if __name__ == '__main__':
    test_InvGamma()
    test_NormInvGamma()
