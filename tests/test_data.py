import numpy as np
import dpmm
from test_utils import timer


@timer
def test_advanced_indexing():
    nobj = 10
    nsample = 9
    ndim = 8

    # actual data
    d = np.empty((nobj, nsample, ndim), dtype=np.float)
    # interim priors
    ips = np.empty((nobj, nsample), dtype=np.float)

    psd = dpmm.PseudoMarginalData(d, ips)

    # compare direct access of data to access through __getitem__
    # 2nd raw item is same as 0th item of [2, 6] fancy indexing
    np.testing.assert_equal(psd.data[2], psd[[2, 6]].data[0],
                            "Advanced indexing didn't work for PseudoMarginalData")
    # Check that the interim prior value works this way too.
    np.testing.assert_equal(psd.interim_prior[2], psd[[2, 6]].interim_prior[0],
                            "Advanced indexing didn't work for PseudoMarginalData")
    # And now check that 6th raw item is 1st item of [2, 6] fancy indexing.
    np.testing.assert_equal(psd.data[6], psd[[2, 6]].data[1],
                            "Advanced indexing didn't work for PseudoMarginalData")
    # and interim prior
    np.testing.assert_equal(psd.interim_prior[6], psd[[2, 6]].interim_prior[1],
                            "Advanced indexing didn't work for PseudoMarginalData")


if __name__ == "__main__":
    test_advanced_indexing()
