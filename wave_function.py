import numpy as np
import numpy.fft as fft
from numba import jit
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from timeit import default_timer

def timer(func):
    def wrapper(*args, **kwargs):
        start = default_timer()
        r = func(*args, **kwargs)
        print("Function {} took {:.4} s".format(func.__name__, default_timer() - start))
        return r
    return wrapper


class ParameterObject:
    def __init__(self, resolutionX = 256, resolutionY = 256,
    x_low = -10, x_high = 10, y_low = -10, y_high = 10,
    beta2 = 1000, omega = 0.8):

        self.resolutionX = resolutionX
        self.resolutionY = resolutionY

        # set the bounddaries of the image box
        self.x_low = x_low
        self.x_high = x_high
        self.y_low = y_low
        self.y_high = y_high

        # calculate the spatial step and make a coordinate array
        self.dx = (self.x_high - self.x_low)/self.resolutionX
        self.x = np.linspace(self.x_low, self.x_high, self.resolutionX)

        self.dy = (self.y_high - self.y_low)/self.resolutionY
        self.y = np.linspace(self.y_low, self.y_high, self.resolutionY)

        # initialize the potential array V

        self.V = np.zeros((self.resolutionX, self.resolutionY))

        # constants for the BEC itself
        self.beta2 = beta2
        self.omega = omega

    def initVharmonic(self, V0 = 1, gamma_y = 1):
        xx, yy = np.meshgrid(self.x, self.y, sparse=False, indexing='xy')
        self.V = V0 * 0.5*(xx**2 + (gamma_y*yy)**2)

    def initVharmonic_quartic(self, alpha=1.3, kappa=0.3):
        xx, yy = np.meshgrid(self.x, self.y, sparse=False, indexing='xy')
        self.V = (1-alpha)*(xx**2 + yy**2) + kappa * (xx**2+yy**2)**2

    def getResolution(self):
        # returns a 2 element tuple with the X- and Y- resolution
        return (self.resolutionX, self.resolutionY)

    def res(self):
        # alias for self.getResolution()  ...  it's just shorter
        return self.getResolution()

    def getBoundaries(self):
        # returns a 4 tuple with the x- and y-boundaries
        return (self.x_low, self.x_high, self.y_low, self.y_high)

class WaveFunction2D:
    def __init__(self, parameterObject):
        if type(parameterObject) != ParameterObject:
            raise TypeError("Argument parameterObject has to be of the type ParameterObject. Given is type {}.".format(type(parameterObject)))
        
        self.paramObj = parameterObject

        self.psi_contains_values = False
        self.psi_array = np.zeros(self.paramObj.getResolution()) + (0+0j)

        self.psi_hat_contains_values = False
        self.psi_hat_array = np.zeros(self.paramObj.getResolution()) + (0+0j)

        self.L_psi_contains_values = False
        self.L_psi_array = np.zeros(self.paramObj.getResolution()) + (0+0j)
    
    def setPsi(self, array):
        # a method to manually set psi to a given 2d array
        if array.shape != self.paramObj.getResolution():
            raise ValueError("Shape {} of input array does not match the reesolution {}.".format(array.shape, self.paramObj.getResolution()))
        self.psi_array = array
        self.psi_contains_values = True

    def setPsiHat(self, array):
        # a method to manually set psi_hat to a given 2d array
        if array.shape != self.paramObj.getResolution():
            raise ValueError("Shape {} of input array does not match the resolution {}.".format(array.shape, self.paramObj.getResolution()))
        self.psi_hat_array = array
        self.psi_hat_contains_values = True

    def initPsiGauss(self, sigma=1, x0=0, y0=0):
        # initializes Psi with a simple 2d gaussian
        xx, yy = np.meshgrid(self.paramObj.x, self.paramObj.y, sparse=False, indexing='xy')

        self.psi_array = 1/(sigma**2) * np.exp(-0.5*((xx-x0)**2 + (yy-y0)**2)/sigma**2)
        self.norm()
        self.psi_contains_values = True
        return self.psi_array

    def initPsi_0(self):
        # initial wave functions according to paper by bao and wang
        def norm(a):
            return a / np.sqrt( np.sum(np.abs(a)**2) * self.paramObj.dx * self.paramObj.dy )

        xx, yy = np.meshgrid(self.paramObj.x, self.paramObj.y, sparse=False, indexing='xy')
        phi_1 = (xx+1j*yy)/np.sqrt(np.pi) * np.exp(-0.5*(xx**2 + yy**2))
        phi_2 = 1/np.sqrt(np.pi) * np.exp(-0.5*(xx**2 + yy**2))
        phi_3 = (phi_1 + phi_2)/2
        phi_3 = norm(phi_3)
        phi_4 = (1-self.paramObj.omega)*phi_2 + self.paramObj.omega * phi_1
        phi_4 = norm(phi_4)

        self.psi_array = phi_4
        self.psi_contains_values = True
        return self.psi_array

    def norm(self):
        # normalizes Psi
        self.psi_array /= np.sqrt( np.sum(np.abs(self.psi_array)**2) * self.paramObj.dx * self.paramObj.dy )
        return self.psi_array

    def getNorm(self):
        # normalizes Psi
        return np.sqrt( np.sum(np.abs(self.psi_array)**2) * self.paramObj.dx * self.paramObj.dy )

    def calcFFT(self):
        # calculates the FFT
        if not self.psi_contains_values:
            raise ValueError("Psi does not contain values or Psi was not initialized!")
        else:
            self.psi_hat_array = np.fft.fft2(self.psi_array) /(np.prod(self.paramObj.getResolution()))
            self.psi_hat_contains_values = True
            return self.psi_hat_array

    def calcIFFT(self):
        # calculates the inverse FFT
        if not self.psi_hat_contains_values:
            raise ValueError("Psi_hat does not contain values or Psi_hat was not initialized!")
        else:
            self.psi_array = np.fft.ifft2(self.psi_hat_array) * (np.prod(self.paramObj.getResolution()))
            self.psi_contains_values = True
            return self.psi_array

    #@timer
    def calcL(self):
        # just a wrapper function for calcL_jit to use a timer
        return self.calcL_jit()

    @jit
    def calcL_jit(self):
        # calculates L acting on psi

        # set some aliases for resolution and boundaries to make the code more readable
        a, b, c, d = self.paramObj.getBoundaries()
        M, N = self.paramObj.getResolution()

        # calculate the FFT of Psi
        if not self.psi_hat_contains_values:
            print("[WARNING] calculating psi_hat since it is empty")
            self.calcFFT()

        # doing this with a for loop because numpy doesnt work the way i want it to
        # Dx_psi = WaveFunction2D(self.paramObj)
        # Dy_psi = WaveFunction2D(self.paramObj)

        # for p in range(-M//2, M//2):
        #     for q in range(-N//2, N//2): 
        #         my_p = 2*p*np.pi/(b-a)
        #         lambda_q = 2*q*np.pi/(d-c)
        #         Dx_psi.psi_hat_array[p,q] = self.psi_hat_array[p,q] * my_p
        #         Dy_psi.psi_hat_array[p,q] = self.psi_hat_array[p,q] * lambda_q
        #         # psi_dx_hat[p,q] = psi_hat[p, q] * my_p
        #         # psi_dy_hat[p,q] = psi_hat[p, q] * lambda_q

        # Dx_psi.psi_hat_contains_values = True
        # Dy_psi.psi_hat_contains_values = True

        # Dx_psi.calcIFFT()
        # #Dx_psi.psi_array /= N*M

        # Dy_psi.calcIFFT()
        # #Dy_psi.psi_array /= N*M

        # calculating D_x(Psi) and D_y(Psi) first to later Calculate L

        p = np.arange(-M//2, M//2, 1)
        q = np.arange(-N//2, N//2, 1)
        pp, qq = np.meshgrid(p, q, indexing='ij')

        lambda_q = 2*qq*np.pi/(d-c)
        my_p = 2*pp*np.pi/(b-a)

        my_p = np.fft.fftshift(my_p)
        lambda_q = np.fft.fftshift(lambda_q)


        Dx_psi = WaveFunction2D(self.paramObj)
        Dx_psi.setPsiHat(my_p * self.psi_hat_array)
        Dx_psi.calcIFFT()
        # Dx_psi.psi_array /= np.prod( M*N )

        Dy_psi = WaveFunction2D(self.paramObj)
        Dy_psi.setPsiHat(lambda_q * self.psi_hat_array)
        Dy_psi.calcIFFT()
        # Dy_psi.psi_array /= np.prod( M*N )

        # adding Dx and Dy up to L
        xx, yy = np.meshgrid(self.paramObj.x, self.paramObj.y, sparse=False, indexing='xy')

        self.L_psi_array = xx * Dy_psi.psi_array - yy * Dx_psi.psi_array
        self.L_psi_contains_values = True
        return self.L_psi_array

    def calcG_m(self, psi_m, alpha):
        # this function calculates G. supposed to be called by psi_n with a given Psi_m as a parameter
        if alpha < 0:
            print("ALERT, alpha < 0!")
        if type(psi_m) != WaveFunction2D:
            raise TypeError("Parameter Psi_m has to be of type WaveFunction2D.")
        if not self.psi_contains_values or not psi_m.psi_contains_values or not psi_m.L_psi_contains_values:
            raise ValueError("Something was not calculated...")
        
        g = alpha * psi_m.psi_array + (0+0j)
        g -= self.paramObj.V * psi_m.psi_array 
        g -= self.paramObj.beta2 * np.abs(self.psi_array)**2 * psi_m.psi_array 
        g += self.paramObj.omega * psi_m.L_psi_array
        return g

    def plot3D(self):
        # just a simple function that quickly plots a wave function
        ax = plt.axes(projection='3d')
        ax.contour3D(self.paramObj.x, self.paramObj.y, np.abs(self.psi_array)**2, 70, cmap='viridis')
        # ax.contour3D(self.paramObj.x, self.paramObj.y, self.psi_array.imag, 70, cmap='viridis')
        plt.show()

