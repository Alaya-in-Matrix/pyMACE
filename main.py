import GPy
import numpy as np
import matplotlib.pyplot as plt
from   GP import GP_MCMC
from   MACE import MACE
import obj
import toml
import multiprocessing
import sys
import os
np.set_printoptions(precision=6, linewidth=500)

# argv = sys.argv[1:]
conf = toml.load("conf.toml")

# TODO: default values for conf
batch_size = conf["batch_size"]
bounds     = np.array(conf["bounds"])
max_iter   = conf["max_iter"]
num_init   = conf["num_init"]
var_name   = conf["var_name"]
use_sobol  = conf["use_sobol"]
warp       = conf["warp"]
mo_eval    = conf["mo_eval"]
mcmc       = conf["mcmc"]

os.system("rm -rf work")
os.system("mkdir work")
for i in range(batch_size):
    copy_cmd = "cp -r ./circuit work/%d" % i
    os.system(copy_cmd)


obj_f  = obj.Obj(bounds, num_init, var_name)
def f(x):
    return obj_f.evaluate(0, x)[0]


dim = len(bounds)
lb  = np.zeros(dim)
ub  = np.zeros(dim)
for i in range(dim):
    lb[i] = bounds[i][0]
    ub[i] = bounds[i][1]

print(f(lb))
print(f(ub))


optimizer = MACE(f, lb, ub, num_init, max_iter, batch_size, sobol_init = use_sobol, warp = warp, mo_eval = mo_eval, mcmc = mcmc);
optimizer.init()
optimizer.optimize()

