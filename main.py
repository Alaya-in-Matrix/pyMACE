import GPy
import numpy as np
import matplotlib.pyplot as plt
from GP import GP_MCMC

num_train = 10;
train_x   = np.random.uniform(-3, 3, (num_train, 1))
train_y   = 100 * np.sin(train_x).reshape(num_train, 1) + np.random.randn(num_train,1)*0.05
gp        = GP_MCMC(train_x, train_y, 4);

xs = np.linspace(-5, 5, 100).reshape(100, 1);

lcbs = xs.copy()
eis  = xs.copy()
pis  = xs.copy()

print("Trained")


for i in range(100):
    x = xs[i];
    lcb, ei, pi = gp.MACE_acq(x);
    print((lcb, ei, pi))
    lcbs[i]     = lcb
    eis[i]      = ei
    pis[i]      = pi

plt.plot(xs, pis)
plt.show()
