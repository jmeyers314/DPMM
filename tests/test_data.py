import numpy as np
import dpmm
from test_utils import timer


@timer
def test_advanced_indexing():
    nobj = 10
    nsample = 9
    ndim = 8

    d = np.empty((nobj, nsample, ndim), dtype=np.float)
    ips = np.empty((nobj, nsample), dtype=np.float)

    psd = dpmm.PseudoMarginalData(d, ips)

    np.testing.assert_equal(psd.data[2], psd[[2, 6]].data[0],
                            "Advanced indexing didn't work for PseudoMarginalData")
    np.testing.assert_equal(psd.interim_post[2], psd[[2, 6]].interim_post[0],
                            "Advanced indexing didn't work for PseudoMarginalData")
    np.testing.assert_equal(psd.data[6], psd[[2, 6]].data[1],
                            "Advanced indexing didn't work for PseudoMarginalData")
    np.testing.assert_equal(psd.interim_post[6], psd[[2, 6]].interim_post[1],
                            "Advanced indexing didn't work for PseudoMarginalData")


if __name__ == "__main__":
    test_advanced_indexing()
