from GP import GP_MCMC
import numpy as np
from platypus import NSGAII, MOEAD, Problem, Real, SPEA2, NSGAIII, Solution, InjectedPopulation, Archive
from math import pow, log, sqrt
from scipy.special import erfc
from scipy.optimize import fmin_l_bfgs_b
from sobol_seq import i4_sobol_generate
import os, sys

class MACE:
    def __init__(self, f, lb, ub, num_init, max_iter, B, debug=True, sobol_init=True, warp = False, mo_eval = 25000, mcmc = True):
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
        self.warp       = warp
        self.mo_eval    = mo_eval
        self.mcmc       = mcmc

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

    def gen_guess(self):
        num_guess     = 1 + len(self.model.ms)
        guess_x       = np.zeros((num_guess, self.dim))
        guess_x[0, :] = self.best_x

        def obj(x, m):
            m, _    = m.predict(x[None, :])
            return m
        def gobj(x, m):
            dmdx, _ = m.predictive_gradients(x[None, :])
            return dmdx

        bounds = [(self.lb[i], self.ub[i]) for i in range(self.dim)]
        for i in range(1, num_guess):
            m  = self.model.ms[i-1]
            xx = self.best_x + np.random.randn(self.best_x.size).reshape(self.best_x.shape) * 1e-3
            def mobj(x):
                return obj(x, m)
            def gmobj(x):
                return gobj(x, m)
            x, _, _ = fmin_l_bfgs_b(mobj, xx, gmobj, bounds=bounds)
            guess_x[i, :] = np.array(x)
        return guess_x



    def optimize(self):
        os.system("rm -f pf* ps* opt.log")
        f           = open('opt.log', 'w');
        self.best_y = np.min(self.dby)
        for iter in range(self.max_iter):
            self.model = GP_MCMC(self.dbx, self.dby, self.B, self.num_init, warp = self.warp, mcmc = self.mcmc)
            print("GP built")
            print(self.model.m, flush=True)

            guess_x   = self.gen_guess()
            num_guess = guess_x.shape[0]

            def obj(x):
                lcb, ei, pi = self.model.MACE_acq(np.array([x]))
                log_ei      = np.log(1e-40 + ei)
                log_pi      = np.log(1e-40 + pi)
                return [lcb[0], -1*log_ei[0], -1*log_pi[0]]

            problem = Problem(self.dim, 3)
            for i in range(self.dim):
                problem.types[i] = Real(self.lb[i], self.ub[i])

            init_s = [Solution(problem) for i in range(num_guess)]
            for i in range(num_guess):
                init_s[i].variables = [x for x in guess_x[i, :]]

            problem.function = obj
            gen              = InjectedPopulation(init_s)
            arch             = Archive()
            algorithm        = NSGAII(problem, population = 100, generator = gen, archive = arch)
            def cb(a):
                print(a.nfe, len(a.archive), flush=True)
            algorithm.run(self.mo_eval, callback=cb)

            if len(algorithm.result) > self.B:
                optimized = algorithm.result
            else:
                optimized = algorithm.population

            idxs = np.arange(len(optimized))
            idxs = np.random.permutation(idxs)
            idxs = idxs[0:self.B]
            for i in idxs:
                x = np.array(optimized[i].variables)
                y = self.f(x)
                if y < self.best_y:
                    self.best_y = y
                    self.best_x = x
                self.dbx = np.concatenate((self.dbx, x.reshape(1, x.size)), axis=0)
                self.dby = np.concatenate((self.dby, y.reshape(1, 1)), axis=0)
            pf       = np.array([s.objectives for s in optimized])
            ps       = np.array([s.variables  for s in optimized])
            self.pf  = pf;
            self.ps  = ps;
            pf[:, 1] = np.exp(-1 * pf[:, 1]) # from -1*log_ei to ei
            pf[:, 2] = np.exp(-1 * pf[:, 2]) # from -1*log_pi to pi
            np.savetxt('pf%d' % iter, pf)
            np.savetxt('ps%d' % iter, ps)
            np.savetxt('dbx', self.dbx)
            np.savetxt('dby', self.dby)

            if self.debug:
                f.write("After iter %d, evaluated: %d, best is %g\n" % (iter, self.dby.size, np.min(self.dby)))
                best_lcb, best_ei, best_pi = self.model.MACE_acq(self.best_x)
                f.write('Best x,  LCB: %g, EI: %g, PI: %g\n' % (best_lcb[0], best_ei[0], best_pi[0]))
                f.write('Tau = %g, eps = %g, kappa = %g, ystd = %g, ymean = %g\n' % (self.model.tau, self.model.eps, self.model.kappa, self.model.std, self.model.mean))
                if self.mcmc:
                    f.write('Hypers:\n' + str(self.model.s)  + '\n')
                evaled_x  = self.dbx[-1*self.B:, :]
                evaled_y  = self.dby[-1*self.B:]
                evaled_pf = self.pf[idxs]

                for i in range(self.B):
                    predy, preds = self.model.predict(evaled_x[i, :]);
                    predy        = predy.reshape(predy.size);
                    preds        = preds.reshape(preds.size);
                    pred         = [(predy[ii], preds[ii]) for ii in range(predy.size)]
                    f.write('X:    ')
                    for d in range(self.dim):
                        f.write(' ' + str(evaled_x[i, d]) + ' ');
                    f.write('\n');
                    f.write('Y:    '   + str(evaled_y[i, 0])      + '\n');
                    f.write('ACQ:  '   + str(evaled_pf[i, :])  + '\n');
                    f.write('Pred:\n'  + str(np.array(pred)) + '\n');
                    f.write('---------------\n')
                f.flush()
        f.close()
