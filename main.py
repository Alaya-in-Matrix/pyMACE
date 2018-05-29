import GPy
import numpy as np
import matplotlib.pyplot as plt
from GP import GP_MCMC
from MACE import MACE

def f(x):
    return x[0]**2 + x[1]**2

optimizer = MACE(f, np.array([-10, -10]), np.array([10, 10]), 4, 10, 4);
optimizer.init()
optimizer.optimize()

