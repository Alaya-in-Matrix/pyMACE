import GPy
import numpy as np
import matplotlib.pyplot as plt
from GP import GP_MCMC

num_train = 20;
train_x   = np.random.uniform(-3, 3, (num_train, 1))
train_y   = 100 * np.sin(train_x).reshape(num_train, 1) + np.random.randn(num_train,1)*0.05
gp        = GP_MCMC(train_x, train_y, 4);
