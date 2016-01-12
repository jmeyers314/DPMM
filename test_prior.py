import numpy as np
import prior

mu_0 = np.r_[0.0, 0.0]
kappa_0 = 2
Lam_0 = np.eye(2)
nu_0 = 2

# Create a Normal-Inverse-Wishart prior.
niw = prior.NIW(mu_0, kappa_0, Lam_0, nu_0)

# Check that we can draw samples from niw.
sample1 = niw.sample()
samples = niw.sample(size=10)

# Check that we can evaluate a likelihood given 1 data point.
theta = (np.r_[1., 1.], np.eye(2)+0.12)
x = np.r_[0.1, 0.2]
lh1_fn = niw.like1(*theta, x=x)
# Or given multiple data points.
D = np.array([[0.1, 0.2], [0.2, 0.3], [0.1, 0.2], [0.4, 0.3]])
print niw.likelihood(*theta, D=D)

# Evaluate prior
print niw(*theta)
print niw.post_params(D)
print niw.pred(x)
print niw.post_pred(D, x)


# Now try GaussianMeanKnownVariance
mu_0 = 0.0
sig_0 = 1.0
sig = 0.1
model = prior.GaussianMeanKnownVariance(mu_0, sig_0, sig)

# Check that we can draw samples from model.
sample1 = model.sample()
samples = model.sample(size=10)
print type(samples)
print type(samples[0])

# Check that we can evaluate a likelihood given 1 data point.
theta = (1.0, )
x = 1.0
lh1_fn = model.like1(*theta, x=x)
# Or given multiple data points.
D = np.array([1.0, 1.0, 1.0, 1.3])
print model.likelihood(*theta, D=D)

# Evaluate prior
print model(*theta)
# Update prior parameters
print model.post_params(D)
# Prior predictive
print model.pred(x)
# Posterior predictive
print model.post_pred(D, x)
