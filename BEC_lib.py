import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
import math
from numba import jit
import pickle
import cmath

pi = math.pi

def save_psi_array(array, filename):
    pickle.dump(array, open("saved_simulations/"+filename, "wb"))

def load_psi_array(filename):
    return pickle.load(open("saved_simulations/"+filename, "rb"))


def display_psi_array(array, playback_speed=20, dynamic_colorbar=True):
    fig = plt.figure()
    shape = array[0].shape
    data = np.random.rand(shape[0],shape[1])
    im = plt.imshow(data, cmap='jet', animated=True)
    plt.colorbar()
    def update(i):
        im.set_data(array[i])
        if dynamic_colorbar:
            vmin = np.min(array[i])
            vmax = np.max(array[i])
            im.set_clim(vmin, vmax)
        return im
    anim = animation.FuncAnimation(fig, update, frames=len(array), interval=playback_speed)
    plt.show()
    return anim

def plot3D(x,y,z):
    ax = plt.axes(projection='3d')
    ax.contour3D(x,y,z, 70, cmap='viridis')
    plt.show()


def potential(x, y, gamma_y):
    return 0.5*(x**2 + (gamma_y*y)**2)


def gaussian(x, y, boundaries, sigma=1, x0=0, y0=0):

    g = 1/(sigma**2 * 2*pi) * np.exp(-0.5*((x-x0)**2 + (y-y0)**2)/sigma**2)
    return g / discrete_norm(g, boundaries)

def discrete_norm(psi, boundaries):
    M, N = psi.shape
    a, b, c, d = boundaries
    dx = (b-a)/M
    dy = (d-c)/N
    return np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)

@jit(nopython=True)
def DFT(psi):
    M, N = psi.shape
    psi_hat = np.zeros((M, N)) + (0+0j)
    for p in range(-M//2, M//2):
        for q in range(-N//2, N//2):
            s = 0
            for j in range(M):
                for k in range(N):
                    s += psi[j,k] * cmath.exp(-1j*2*j*p*pi/M) * cmath.exp(-1j*2*k*q*pi/N)
            psi_hat[p,q] = 1/(M*N)
            psi_hat[p,q] *= s
    return psi_hat

def fastDFT(psi):
    M, N = psi.shape
    return np.fft.fft2(psi)/(M*N)

@jit(nopython=True)
def iDFT(psi):
    M, N = psi.shape
    psi_hat = np.zeros((N, M)) + (0+0j)
    for p in range(-M//2, M//2):
        for q in range(-N//2, N//2):
            s = 0
            for j in range(M):
                for k in range(N):
                    s += psi[j,k] * cmath.exp(1j*2*j*p*pi/M) * cmath.exp(1j*2*k*q*pi/N)
            psi_hat[p,q] = s
    return psi_hat

def fastiDFT(psi):
    M, N = psi.shape
    return np.fft.ifft2(psi)*N*M


@jit(nopython=True)
def nabla_2(psi_hat, boundaries):
    a, b, c, d = boundaries
    M, N = psi_hat.shape
    psi = np.ones((M,N)) + (0+0j)
    for j in range(M):
        for k in range(N):
            s = 0
            for p in range(-M//2, M//2):
                for q in range(-N//2, N//2):
                    my_p = 2*p*pi/(b-a)
                    lambda_q = 2*q*pi/(d-c)

                    s += (my_p**2 + lambda_q**2) * psi_hat[p, q] * cmath.exp(-1j*2*j*p*pi/M) * cmath.exp(-1j*2*k*q*pi/N)
            psi[j,k] = s
    return -psi

@jit
def L(psi, psi_hat, boundaries):
    a, b, c, d = boundaries
    M, N = psi.shape
    psi_dx = np.ones((M,N)) + (0+0j)
    psi_dy = np.ones((M,N)) + (0+0j)
    psi_L = np.ones((M,N)) + (0+0j)
    for j in range(M):
        for k in range(N):
            s_dx = 0
            s_dy = 0
            for p in range(-M//2, M//2):
                for q in range(-N//2, N//2):
                    my_p = 2*p*pi/(b-a)
                    lambda_q = 2*q*pi/(d-c)
                    tmp = psi_hat[p, q] * cmath.exp(-1j*2*j*p*pi/M) * cmath.exp(-1j*2*k*q*pi/N)
                    s_dx += my_p * tmp
                    s_dy += lambda_q * tmp
            psi_dx[j,k] = s_dx
            psi_dy[j,k] = s_dy
    #bad!
    dx = (b-a)/M
    x = np.arange(a, b, dx)
    dy = (d-c)/N
    y = np.arange(c, d, dy)
    xx, yy = np.meshgrid(x, y, sparse=False, indexing='xy')

    psi_L = xx * psi_dy - yy * psi_dx
    return psi_L

@jit
def fastL(psi, psi_hat, boundaries):
    a, b, c, d = boundaries
    M, N = psi.shape

    p = np.arange(-M//2, M//2, 1)
    q = np.arange(-N//2, N//2, 1)
    pp, qq = np.meshgrid(p, q, indexing='xy')

    lambda_q = 2*qq*pi/(d-c)
    my_p = 2*pp*pi/(b-a)

    psi_dx_hat = my_p * psi_hat
    psi_dy_hat = lambda_q * psi_hat
        
    psi_dx = fastDFT(psi_dx_hat)
    psi_dy = fastDFT(psi_dy_hat)

    # adding Dx and Dy up to L

    dx = (b-a)/M
    x = np.arange(a, b, dx)
    dy = (d-c)/N
    y = np.arange(c, d, dy)
    xx, yy = np.meshgrid(x, y, sparse=False, indexing='xy')

    psi_L = xx * psi_dy - yy * psi_dx
    return psi_L

@jit(nopython=True)
def G(psi_n, psi_m, psi_m_L, V, beta, omega):
    b = V + beta*np.abs(psi_n)**2
    bmin = np.min(b)
    bmax = np.max(b)
    alpha = 0.5 * (bmax + bmin)
    if alpha < 0:
        print("ALERT, alpha < 0!")

    g = alpha * psi_m - V * psi_m - beta*np.abs(psi_n)**2 * psi_m + omega * psi_m_L
    return g

@jit(nopython=True)
def next_psi(psi_hat, g_hat, dt, alpha, boundaries):
    a, b, c, d = boundaries
    M, N = psi_hat.shape
    next_psi_hat = np.zeros((N,M)) + (0+0j)
    for p in range(-M//2, M//2):
        for q in range(-N//2, N//2):
            my_p = 2*p*pi/(b-a)
            lambda_q = 2*q*pi/(d-c)
            next_psi_hat[p,q] = psi_hat[p,q] + dt * g_hat[p,q]
            next_psi_hat[p,q] *= 2 / (2 + dt*(2*alpha + my_p**2 + lambda_q**2))
    return next_psi_hat