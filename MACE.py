from GP import GP_MCMC
import numpy as np
from platypus import NSGAII, MOEAD, Problem, Real, SPEA2, NSGAIII, Solution, InjectedPopulation
from math import pow, log, sqrt
from scipy.special import erfc
from sobol_seq import i4_sobol_generate
import os

class MACE:
    def __init__(self, f, lb, ub, num_init, max_iter, B, debug=True, sobol_init=True):
        """
        f: the objective function:
            input: D row vector
            output: scalar value
        lb: lower bound
        ub: upper bound
        num_init: number of initial random sampling
        max_iter: number of iterations
        B: batch size, the total number of function evaluations would be num_init + B * max_iter
        """
        self.f          = f
        self.lb         = lb.reshape(lb.size)
        self.ub         = ub.reshape(ub.size)
        self.dim        = self.lb.size
        self.num_init   = num_init
        self.max_iter   = max_iter
        self.B          = B
        self.debug      = debug
        self.sobol_init = sobol_init

    def init(self):
        self.dbx = np.zeros((self.num_init, self.dim))
        if self.sobol_init:
            self.dbx = (self.ub - self.lb) * i4_sobol_generate(self.dim, self.num_init) +  self.lb
        else:
            self.dbx = np.random.uniform(self.lb, self.ub, (self.num_init, self.dim))

        self.dby    = np.zeros((self.num_init, 1))
        self.best_y = np.inf
        for i in range(self.num_init):
            y = self.f(self.dbx[i])
            if y < self.best_y:
                self.best_y = y
                self.best_x = self.dbx[i]
            self.dby[i] = y;
        np.savetxt('dbx', self.dbx)
        np.savetxt('dby', self.dby)
        print('Initialized, best is %g' % self.best_y)

    def optimize(self):
        os.system("rm -f pf* ps* opt.log")
        f           = open('opt.log', 'w');
        self.best_y = np.min(self.dby)
        for iter in range(self.max_iter):
            self.model = GP_MCMC(self.dbx, self.dby, self.B, self.num_init)
            print("GP built")
            print(self.model.m)

            def obj(x):
                lcb, ei, pi = self.model.MACE_acq(np.array([x]))
                log_ei      = np.log(1e-40 + ei)
                log_pi      = np.log(1e-40 + pi)
                return [lcb[0], -1*log_ei[0], -1*log_pi[0]]

            problem = Problem(self.dim, 3)
            for i in range(self.dim):
                problem.types[i] = Real(self.lb[i], self.ub[i])

            # The current best solution as an initial guess of the NSGAIII population
            s1 = Solution(problem);
            s2 = Solution(problem);
            s3 = Solution(problem);
            for i in range(self.dim):
                s1.variables[i] = np.maximum(self.lb[i], np.minimum(self.ub[i], self.best_x[i]))
                s2.variables[i] = np.maximum(self.lb[i], np.minimum(self.ub[i], 1e-3 * np.random.randn() + self.best_x[i]))
                s3.variables[i] = np.maximum(self.lb[i], np.minimum(self.ub[i], 1e-3 * np.random.randn() + self.dbx[-1, i]))

            problem.function = obj
            gen              = InjectedPopulation([s1, s2, s3])
            # algorithm        = MOEAD(problem, population_size=100, generator = gen)
            algorithm        = NSGAIII(problem, divisions_outer=12, generator = gen)
            # algorithm        = NSGAII(problem, generator = gen)
            # algorithm        = CMAES(problem, epsilons=0.05, population_size=100, generator = gen)
            algorithm.run(25000)

            idxs = np.random.randint(0, len(algorithm.result), self.B)
            for i in idxs:
                x = np.array(algorithm.result[i].variables)
                y = self.f(x)
                if y < self.best_y:
                    self.best_y = y
                    self.best_x = x
                self.dbx = np.concatenate((self.dbx, x.reshape(1, x.size)), axis=0)
                self.dby = np.concatenate((self.dby, y.reshape(1, 1)), axis=0)
            f.write("Iter %d, evaluated: %d, best is %g\n" % (iter, self.dby.size, np.min(self.dby)))
            pf       = np.array([s.objectives for s in algorithm.result])
            ps       = np.array([s.variables  for s in algorithm.result])
            self.pf  = pf;
            self.ps  = ps;
            pf[:, 1] = np.exp(-1 * pf[:, 1]) # from -1*log_ei to ei
            pf[:, 2] = np.exp(-1 * pf[:, 2]) # from -1*log_pi to pi
            np.savetxt('pf%d' % iter, pf)
            np.savetxt('ps%d' % iter, ps)
            np.savetxt('dbx', self.dbx)
            np.savetxt('dby', self.dby)

            if self.debug:
                best_lcb, best_ei, best_pi = self.model.MACE_acq(self.best_x)
                f.write('MAP model:\n%s\n' % str(self.model.m))
                f.write('Best x,  LCB: %g, EI: %g, PI: %g\n' % (best_lcb[0], best_ei[0], best_pi[0]))
                f.write('Tau = %g, eps = %g, kappa = %g, ystd = %g, ymean = %g\n' % (self.model.tau, self.model.eps, self.model.kappa, self.model.std, self.model.mean))
                for i in range(len(ps)):
                    x      = ps[i, :]
                    fx     = self.f(x)
                    f.write('True value: %g\n' % fx)
                    acq    = pf[i, :]
                    predy, preds = self.model.predict(x)
                    f.write('PY:  ' + str(predy.reshape(predy.size)) + '\n')
                    f.write('PS:  ' + str(preds.reshape(preds.size)) + '\n')
                    f.write('ACQ: ' + str(acq)                       + '\n')
                    f.write('---------------\n')
        f.close()
