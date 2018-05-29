from GP import GP_MCMC
import numpy as np
from platypus import NSGAII, Problem, Real, SPEA2, NSGAIII

class MACE:
    def __init__(self, f, lb, ub, num_init, max_iter, B):
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
    def init(self):
        self.dbx = np.zeros((self.num_init, self.dim))
        self.dby = np.zeros((self.num_init, 1))
        # TODO: the initialization can be paralleled
        for i in range(self.num_init):
            x           = np.random.uniform(self.lb, self.ub).reshape(self.dim)
            y           = self.f(x)
            self.dbx[i] = x;
            self.dby[i] = y;
        print('Initialized, best is %g' % np.min(self.dby))

    def optimize(self):
        for iter in range(self.max_iter):
            self.model = GP_MCMC(self.dbx, self.dby, self.B)
            print("GP built")
            print(self.model.m)

            def obj(x):
                lcb, ei, pi = self.model.MACE_acq(np.array([x]))
                return [lcb[0], -1*ei[0], -1*pi[0]]

            problem = Problem(self.dim, 3)
            for i in range(self.dim):
                problem.types[i] = Real(self.lb[i], self.ub[i])
            problem.function = obj
            algorithm        = NSGAII(problem, population_size=100)
            algorithm.run(10000)

            idxs = np.random.randint(0, len(algorithm.result), self.B)
            for i in idxs:
                x = np.array(algorithm.result[i].variables)
                y = self.f(x)
                self.dbx = np.concatenate((self.dbx, x.reshape(1, x.size)), axis=0)
                self.dby = np.concatenate((self.dby, y.reshape(1, 1)), axis=0)
            print("iter %d, evaluated: %d, best is %g" % (iter, self.dby.size, np.min(self.dby)))
            pf = np.array([s.objectives for s in algorithm.result])
            ps = np.array([s.variables  for s in algorithm.result])
            np.savetxt('pf', pf)
            np.savetxt('ps', ps)
            np.savetxt('dbx', self.dbx)
            np.savetxt('dby', self.dby)
