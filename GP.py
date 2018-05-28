import GPy
import numpy as np

class GP:
    def __init__(self, train_x, train_y):
        self.train_x   = train_x.copy()
        self.train_y   = train_y.reshape(train_y.size, 1).copy()
        self.num_train = self.train_x.shape[0]
        self.dim       = self.train_x.shape[1]
        kern           = GPy.kern.Matern52(input_dim = self.dim, ARD = True)
        self.m         = GPy.models.GPRegression(X, Y, kern)

        self.m.kern.variance.set_prior(GPy.priors.Gamma.from_EV(np.var(self.train_y), 10.))
        self.m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1e-2 * np.var(self.train_y), 10))
        self.m.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(np.std(self.train_x), 10 * np.ones(np.std(self.train_x).shape)))

        self.eps     = 1e-3;
        self.upsilon = 0.5;
        self.delta   = 0.05
        self.tau     = np.min(self.train_y)
        
    def sample(self):
        hmc    = GPy.inference.mcmc.HMC(self.m,stepsize=5e-2)
        s      = hmc.sample(num_samples=200) # Burnin
        s      = hmc.sample(num_samples=200)
        self.s = s.copy()
        # self.sample_variance = s[:, 0];
        # self.sample_len      = s[:, 1:1+self.dim];
        # self.sample_noise    = s[:, 1+self.dim];
        pass

    def predict(self, x, hyp_vec):
        pass

    def EI(self):
        pass

    def LCB(self):
        pass

    def PI(self):
        pass

    def MACE_acq(self):
        pass
