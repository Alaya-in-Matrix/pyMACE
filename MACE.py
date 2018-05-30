from GP import GP_MCMC
import numpy as np
from platypus import NSGAII, MOEAD, Problem, Real, SPEA2, NSGAIII
from math import pow, log, sqrt
import os

class MACE:
    def __init__(self, f, lb, ub, num_init, max_iter, B, debug=True):
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
        self.f        = f
        self.lb       = lb.reshape(lb.size)
        self.ub       = ub.reshape(ub.size)
        self.dim      = self.lb.size
        self.num_init = num_init
        self.max_iter = max_iter
        self.B        = B
        self.debug    = debug

    def init(self):
        self.dbx = np.zeros((self.num_init, self.dim))
        self.dby = np.zeros((self.num_init, 1))
        # TODO: the initialization can be paralleled
        self.best_y = np.inf
        for i in range(self.num_init):
            x           = np.random.uniform(self.lb, self.ub).reshape(self.dim)
            y           = self.f(x)
            if y < self.best_y:
                self.best_y = y
                self.best_x = x
            self.dbx[i] = x;
            self.dby[i] = y;
        print('Initialized, best is %g' % np.min(self.dby))

    def optimize(self):
        os.system("rm -f dbx dby pf* ps* opt.log")
        f           = open('opt.log', 'w');
        self.best_y = np.min(self.dby)
        for iter in range(self.max_iter):
            self.model = GP_MCMC(self.dbx, self.dby, self.B, self.num_init)
            print("GP built")
            print(self.model.m)

            def obj(x):
                lcb, ei, pi = self.model.MACE_acq(np.array([x]))
                return [lcb[0], -1*ei[0], -1*pi[0]]

            problem = Problem(self.dim, 3)
            for i in range(self.dim):
                problem.types[i] = Real(self.lb[i], self.ub[i])
            problem.function = obj
            # algorithm        = MOEAD(problem, population_size=100)
            algorithm        = NSGAIII(problem, divisions_outer=12, population_size=100)
            # algorithm        = CMAES(problem, epsilons=0.05, population_size=100)
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
            pf = np.array([s.objectives for s in algorithm.result])
            ps = np.array([s.variables  for s in algorithm.result])
            np.savetxt('pf%d' % iter, pf)
            np.savetxt('ps%d' % iter, ps)
            np.savetxt('dbx', self.dbx)
            np.savetxt('dby', self.dby)

            if self.debug:
                f.write('MAP model:\n%s\n' % str(self.model.m))
                f.write('Best x,  LCB: %g, EI: %g, PI: %g\n' % (best_lcb[0], best_ei[0], best_pi[0]))
                f.write('Tau = %g, eps = %g, kappa = %g, ystd = %g, ymean = %g\n' % (self.model.tau, self.model.eps, self.model.kappa, self.model.std, self.model.mean))
                best_lcb, best_ei, best_pi = self.model.MACE_acq(self.best_x)
                for i in range(len(ps)):
                    x      = ps[i, :]
                    fx     = self.f(x)
                    f.write('True value: %g\n' % fx)
                    acq    = pf[i, :]
                    predy, preds = self.model.predict(x)
                    f.write('PY: ' + str(predy.reshape(predy.size)) + '\n')
                    f.write('PS: ' + str(preds.reshape(preds.size)) + '\n')
                    f.write('ACQ:' + str(acq)                       + '\n')
                    f.write('---------------\n')
        f.close()
