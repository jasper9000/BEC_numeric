import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
import math
import cmath
import numpy.fft as fft
from numba import jit
import pickle
import os
import time

from BEC_lib import *

pi = math.pi

def calculate_time_step(psi_n, V, boundaries, omega, beta2, epsilon_limit=10e-6):
    # set up epsilon
    psi_max = 0
    psi_max_new = np.max(psi_n)
    epsilon = 1
    m = 1

    # calculate alpha
    b_ = V + beta2*np.abs(psi_n)**2
    bmin = np.min(b_)
    bmax = np.max(b_)
    alpha = 0.5 * (bmax + bmin)

    # set up initial values for wave functions
    psi_n_hat = fastDFT(psi_n)
    psi_m = psi_n
    psi_m_hat = psi_n_hat
    psi_m_L = fastL(psi_m, psi_m_hat, boundaries)
    G_m = G(psi_n, psi_m, psi_m_L, V, beta2, omega)
    G_m_hat = fastDFT(G_m)

    while epsilon > epsilon_limit:

        psi_m_hat = next_psi(psi_n_hat, G_m_hat, dt, alpha, boundaries)
        psi_m = fastiDFT(psi_m_hat)
        psi_m_L = fastL(psi_m, psi_m_hat, boundaries)

        G_m = G(psi_n, psi_m, psi_m_L, V, beta2, omega)
        G_m_hat = fastDFT(G_m)

        # calculate epsilon
        psi_max = psi_max_new
        psi_max_new = np.max(psi_m)
        epsilon = abs(psi_max - psi_max_new)

        # next step
        
        m += 1

    # end of iteration
    # renormalize psi_m for next time step (part of imaginary time method)
    return psi_m / discrete_norm(psi_m, boundaries)


# mesh size
N = 2 ** 8
M = 2 ** 8

#boundaries
a = -10
b = 10
dx = (b-a)/M
x = np.arange(a, b, dx)

c = -10
d = 10
dy = (d-c)/N
y = np.arange(c, d, dy)

boundaries = (a, b, c, d)


###### constants #######
beta2 = 8000
omega = 100

# potential
V0 = 5
wx = 10
wy = 15
gamma_y = wy / wx

# psi_0
sigma = 0.3
x0 = 0
y0 = 0

# time propagation
dt = 0.005
epsilon_t_limit = 10e-7
epsilon_m_limit = 10e-7


##########

# make potential and psi_0 arrays
# use meshgrid
xx, yy = np.meshgrid(x, y, sparse=False, indexing='xy')

V = V0 * potential(xx, yy, gamma_y)
psi_0 = gaussian(xx, yy, boundaries, sigma, x0, y0)



# start of program execution
psi_n = psi_0
psi_dt_array = [psi_n]

# set up epsilon_t
psi_max = 0
psi_max_new = np.max(psi_n)
epsilon_t = 1
t = 1

while epsilon_t > epsilon_t_limit:
    # do the time step
    psi_n = calculate_time_step(psi_n, V, boundaries, omega, beta2, epsilon_m_limit)
    psi_dt_array.append( psi_n )

    # calculate epsilon
    psi_max = psi_max_new
    psi_max_new = np.max(psi_n)
    epsilon_t = abs(psi_max - psi_max_new)

    print("t = {}, epsilon_t = {}".format(t, epsilon_t))
    t += 1



filename = "dt_" + str(dt) + "_beta_" + str(int(beta2)) + "_omega_" + str(int(omega)) + "_" + str(int(time.time())) + ".p"
# save_psi_array(psi_dt_array, "saved_simulations/"+filename)
display_psi_array(np.abs(psi_dt_array)**2)
