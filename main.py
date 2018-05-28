import GPy
import numpy as np
import matplotlib.pyplot as plt

X    = np.random.uniform(-3, 3, (20, 2))
Y    = 100 * np.sin(X[:, 0] + X[:, 1]) + np.random.randn(20,1)*0.05
kern = GPy.kern.Matern52(input_dim = X.shape[1], variance = np.var(Y), ARD = True, lengthscale = np.std(X, 0))
m    = GPy.models.GPRegression(X, Y, kern)
m.likelihood.variance = 1e-2 * np.var(Y)

m.kern.variance.set_prior(GPy.priors.Gamma.from_EV(np.var(Y), 10.))
m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1e-2 * np.var(Y), 10.))
m.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(np.std(X), 10 * np.ones(np.std(X).shape)))

# print(m)
# print(m.kern)
# print(m.likelihood)

hmc = GPy.inference.mcmc.HMC(m,stepsize=5e-2)
s   = hmc.sample(num_samples=50) # Burnin
s   = hmc.sample(num_samples=50)
print(s)

# print(m.kern.lengthscale)
# m.optimize_restarts(num_restarts = 10)
# print(m.kern.lengthscale)
# fig = m.plot()
# GPy.plotting.show(fig, filename='basic_gp_regression_notebook')

