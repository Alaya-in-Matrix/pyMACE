import GPy
from GPyOpt.util.general import get_quantiles
import numpy as np
from math import pow, log, sqrt

# TODO: standardize the training data
class GP_MCMC:
    def __init__(self, train_x, train_y, B):
        self.train_x   = train_x.copy()
        self.train_y   = train_y.reshape(train_y.size, 1).copy()
        self.num_train = self.train_x.shape[0]
        self.dim       = self.train_x.shape[1]
        self.B         = B

        kern           = GPy.kern.Matern52(input_dim = self.dim, ARD = True)
        self.m         = GPy.models.GPRegression(self.train_x, self.train_y, kern)

        self.m.kern.variance.set_prior(GPy.priors.Gamma.from_EV(np.var(self.train_y), 10.))
        self.m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1e-2 * np.var(self.train_y), 10))
        self.m.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(np.std(self.train_x), 10 * np.ones(np.std(self.train_x).shape)))

        self.eps     = 1e-3;
        self.upsilon = 0.5;
        self.delta   = 0.05
        self.tau     = np.min(self.train_y)

        self.sample()
        
    def sample(self):
        hmc    = GPy.inference.mcmc.HMC(self.m,stepsize=5e-2)
        s      = hmc.sample(num_samples=100) # Burnin
        s      = hmc.sample(num_samples=100)
        self.s = s.copy()

    def predict(self, x, hyp_vec):
        self.m.kern[:] = hyp_vec
        py, ps2        = self.m.predict(x)
        return py, ps2;

    def set_kappa(self):
        t = 1 + int(self.num_train / self.B)
        self.kappa = sqrt(self.upsilon * 2 * log(pow(t, 2.0 + self.dim / 2.0) * 3 * pow(np.pi, 2) / (3 * self.delta)));

    def LCB(self, x):
        self.set_kappa()
        acq = 0;
        for hyp in self.s:
            y, s = self.predict(x, hyp)
            lcb  = y - self.kappa * np.sqrt(s)
            acq += lcb
        acq /= self.s.shape[0]
        return acq

    def EI(self, x):
        self.set_kappa()
        acq = 0;
        for hyp in self.s:
            y, s         = self.predict(x, hyp)
            phi, Phi, u  = get_quantiles(self.eps, self.tau, y, s)
            f_acqu       = s * (u * Phi + phi)
            acq         += f_acqu
        acq /= self.s.shape[0]
        return acq

    def PI(self):
        self.set_kappa()
        acq = 0;
        for hyp in self.s:
            y, s         = self.predict(x, hyp)
            phi, Phi, u  = get_quantiles(self.eps, self.tau, y, s)
            f_acqu       = Phi
            acq         += f_acqu
        acq /= self.s.shape[0]
        return acq

    def MACE_acq(self, x):
        lcb = self.LCB(x)
        ei  = self.ei(x)
        pi  = self.pi(x)
        return lcb, ei, pi
