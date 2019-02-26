__author__ = "Jasper Riebesehl"
__version__ = "1.0"
__email__ = "jasper.riebesehl@physnet.uni-hamburg.de"

# References used in comments:
# [1] Zeng, R & Zhang, Yanzhi. (2009). Efficiently computing vortex lattices in rapid rotating Bose–Einstein condensates.
#     Computer Physics Communications. 180. 854-860. 10.1016/j.cpc.2008.12.003. 
#
# [2] Weizhu Bao, I-Liang Chern & Fong Yin Lim. (2006). Efficient and spectrally accurate numerical methods for computing ground
#     and first excited states in Bose–Einstein condensates. Journal of Computational Physics. Volume 219, Issue 2.
#     https://doi.org/10.1016/j.jcp.2006.04.019.

import numpy as np
import numpy.fft as fft
from numba import jit

from .parameter_object import ParameterObject, Psi0Choice

# implementation of the simpson integration rule for the calculation of the observables
def simpson(y, x):
    '''Simpson rule for any amount of intervals.
    Only works for equal intervals.
    '''
    N = len(x)
    if N%2==1:
        s = simpson_even(y, x)
    else:
        s = simpson_even(y[:-1], x[:-1])
        # trapeziod rule for last iterval
        s += (x[-1]-x[-2]) * (y[-1]+y[-2])/2
    return s

@jit
def simpson_even(y, x):
    '''Simpson rule for even amount of intervals.
    Only works for equal intervals.
    '''
    N = len(x)
    a = x[0]
    b = x[-1]
    h = (b-a)/(N-1)
    
    s = 2*np.sum(y[2:-1:2]) + 4*np.sum(y[1:-1:2])
    s += y[0] + y[-1]
    return s * h/3


class WaveFunction2D:
    '''This class represents a complex wave function in 2 dimensions and contains all neccessary functions to manipulate it.
    Especially this class also contains functionality for the fourier transform psi_hat of the wave function and several other "components".

    Attributes:
        paramObj: A ParameterObject instance.
        psi_array: A 2D numpy array that contains complex values of the wavefunction.
        psi_contains_values: A boolean that indicates whether psi_array was initialized.
        psi_hat_array: A 2D numpy array that contains complex values of the fourier transform of the wavefunction.
        psi_hat_contains_values: A boolean that indicates whether psi_hat_array was initialized.
        L_psi_array: A 2D numpy array that contains complex values of the angular momentum operator applied to the wavefunction.
        L_psi_contains_values: A boolean that indicates whether L_psi_array was initialized.
        nabla_psi_array: A 2D numpy array that contains complex values of the Nabla operator applied to the wavefunction.
        nabla_psi_contains_values: A boolean that indicates whether nabla_psi_array was initialized.
        E: The energy expectation value of the wavefunction.
        L_expectation: The angular momentum expectation value of the wavefunction.
        Nabla_expectation: The kinetic energy expectation value of the wavefunction.
    '''
    def __init__(self, parameterObject):
        """Initializes the instance. Sets all 2D arrays to zero."""
        if type(parameterObject) != ParameterObject:
            raise TypeError("Argument parameterObject has to be of the type {}. Given is type {}.".format(type(ParameterObject), type(parameterObject)))
        
        self.paramObj = parameterObject

        self.psi_contains_values = False
        self.psi_array = np.zeros(self.paramObj.getResolution()) + (0+0j)

        self.psi_hat_contains_values = False
        self.psi_hat_array = np.zeros(self.paramObj.getResolution()) + (0+0j)

        self.L_psi_contains_values = False
        self.L_psi_array = np.zeros(self.paramObj.getResolution()) + (0+0j)

        self.nabla_psi_contains_values = False
        self.nabla_psi_array = np.zeros(self.paramObj.getResolution()) + (0+0j)

        self.E = None
        self.L_expectation = None
        self.Nabla_expectation = None
    
    def setPsi(self, array):
        """A method to manually set psi_array to a given 2d array.

        Arguments:
            array: A 2D (complex) numpy array that will overwrite psi_array.
        """
        # checks if the shape of the array matches the reolution
        if array.shape != self.paramObj.getResolution():
            raise ValueError("Shape {} of input array does not match the resolution {}.".format(array.shape, self.paramObj.getResolution()))
        self.psi_array = array
        self.psi_contains_values = True

    def setPsiHat(self, array):
        """A method to manually set psi_hat_array to a given 2d array.

        Arguments:
            array: A 2D (complex) numpy array that will overwrite psi_hat_array.
        """
        # checks if the shape of the array matches the reolution
        if array.shape != self.paramObj.getResolution():
            raise ValueError("Shape {} of input array does not match the resolution {}.".format(array.shape, self.paramObj.getResolution()))
        self.psi_hat_array = array
        self.psi_hat_contains_values = True

    def initPsi_0(self):
        """ Sets psi_arrays to the correct form.
        Depending on the setting of psi0_choice, psi_array is either a Gauss curve or 
        the Thomas-Fermi-Approximation.
        """
        if self.paramObj.psi0_choice == Psi0Choice.THOMAS_FERMI:
            self.initThomasFermi(self.paramObj.psi0_parameters['gamma_y'])
        elif self.paramObj.psi0_choice == Psi0Choice.GAUSS:
            self.initPsiGauss(self.paramObj.psi0_parameters['sigma'],
            self.paramObj.psi0_parameters['x0'],
            self.paramObj.psi0_parameters['y0'])
        else:
            raise ValueError("Psi0 choice not recognized.")

    def initThomasFermi(self, gamma_y):
        """Sets psi_array to the Thomas-Fermi-Approximation.

        Arguments:
            gamma_y: Value greater than 0 that controls the anisotropy of the function.
        """ 
        my_g = 0.5*np.sqrt(4*self.paramObj.beta2*gamma_y)
        self.psi_array = np.sqrt(np.maximum(0, my_g - self.paramObj.V)/self.paramObj.beta2)
        self.norm()
        self.psi_contains_values = True

    def initPsiGauss(self, sigma=1, x0=0, y0=0):
        """Initializes psi_array with a simple 2d gaussian.

        Arguments:
            sigma: the standart deviation
            x0: x coordinate of the centerpoint
            y0: y coordinate of the centerpoint
        """
        xx, yy = np.meshgrid(self.paramObj.x, self.paramObj.y, sparse=False, indexing='ij')

        self.psi_array = 1/(sigma**2) * np.exp(-0.5*((xx-x0)**2 + (yy-y0)**2)/sigma**2)
        self.norm()
        self.psi_contains_values = True

    def initPsiGauss_double(self, sigma=1, x0=2, y0=0):
        """ initializes Psi with a double 2d gaussian
        Uses the normal distribution formula 2 times to initialize Psi, also normalizes it.
        This function is not used in the GUI and just for fun.

        Args:
            sigma: the standart deviation
            x0: x coordinate of the centerpoint
            y0: y coordinate of the centerpoint
        """
        xx, yy = np.meshgrid(self.paramObj.x, self.paramObj.y, sparse=False, indexing='ij')
        self.psi_array = 1/(sigma**2) * np.exp(-0.5*((xx-x0)**2 + (yy-y0)**2)/sigma**2)
        self.psi_array += 1/(sigma**2) * np.exp(-0.5*((xx+x0)**2 + (yy+y0)**2)/sigma**2)
        self.norm()
        self.psi_contains_values = True

    def norm(self):
        """Normalizes psi_array.
        Uses the normailization formula from [1].
        """
        self.psi_array /= np.sqrt( np.sum(np.abs(self.psi_array[1:-1, 1:-1])**2) * self.paramObj.dx * self.paramObj.dy )
        return self.psi_array

    def getNorm(self):
        """Returns the norm of psi_array.
        """
        return np.sqrt( np.sum(np.abs(self.psi_array[1:-1, 1:-1])**2) * self.paramObj.dx * self.paramObj.dy )

    def calcFFT(self):
        """Calculates and saves the fast Fourier Tranform (FFT) of psi_array.
        """
        # checks if psi_array was initialized
        if not self.psi_contains_values:
            raise ValueError("Psi does not contain values or Psi was not initialized!")
        else:
            # uses the numpy fft for speed since this operation has to be performed a lot.
            self.psi_hat_array = np.fft.fft2(self.psi_array) /(np.prod(self.paramObj.getResolution()))
            self.psi_hat_contains_values = True

    def calcIFFT(self):
        """Calculates and saves the inverse fast Fourier Tranform (IFFT) of psi_array.
        """
        # checks if psi_hat_array was initialized
        if not self.psi_hat_contains_values:
            raise ValueError("Psi_hat does not contain values or Psi_hat was not initialized!")
        else:
            # uses the numpy fft for speed since this operation has to be performed a lot.
            self.psi_array = np.fft.ifft2(self.psi_hat_array) * (np.prod(self.paramObj.getResolution()))
            self.psi_contains_values = True

    def calcEnergy(self):
        """Calulates the expectaion value for the total (GPE) energy.

        Returns:
            Expectation value for the energy.
        """
        # set up aliases
        x = self.paramObj.x
        y = self.paramObj.y

        # add each term of the GPE energy [1] to avoid mistakes
        dE = 0.5*np.abs(self.nabla_psi_array)**2 
        dE += self.paramObj.V * np.abs(self.psi_array)**2
        dE += self.paramObj.beta2/2 * np.abs(self.psi_array)**4
        dE -= self.paramObj.omega * (np.conjugate(self.psi_array)*self.L_psi_array).real

        # 2d integrate by using the simpson rule twice on different axes
        self.E = simpson(simpson(dE, y), x)
        return self.E

    def calcL_expectation(self):
        """Calculates the expectation value for the angular momentum.

        Returns:
            Expectation value for the angular momentum.
        """
        # set up aliases
        x = self.paramObj.x
        y = self.paramObj.y

        # set up integrant
        dL = np.conjugate(self.psi_array)*self.L_psi_array

        # 2d integrate by using the simpson rule twice on different axes
        self.L_expectation = simpson(simpson(dL, y), x)
        return self.L_expectation

    def calcNabla_expectation(self):
        '''Calculates the expectation value for the kinetic energy.
        '''
        # set up aliases
        x = self.paramObj.x
        y = self.paramObj.y

        # set up integrant
        dN = 0.5*np.abs(self.nabla_psi_array)**2

        # 2d integrate by using the simpson rule twice on different axes
        self.Nabla_expectation = simpson(simpson(dN, y), x)
        return self.Nabla_expectation

    def calcNabla(self):
        """Calculates the spatial derivative operator nabla applied to the wavefunction.
        Retruns:
            The Wavefunction after the nabla operator was applied.
        """
        # set up aliases
        a, b, c, d = self.paramObj.getBoundaries()
        M, N = self.paramObj.getResolution()
        
        # set up grid in fourier space
        p = np.arange(-M//2, M//2, 1)
        q = np.arange(-N//2, N//2, 1)
        pp, qq = np.meshgrid(p, q, indexing='ij')

        # set up 'derivation' constants in fourier space
        lambda_q = 2*qq*np.pi/(d-c)
        my_p = 2*pp*np.pi/(b-a)

        # shift 'derivation' constants, will give wrong results otherwise
        my_p = np.fft.fftshift(my_p)
        lambda_q = np.fft.fftshift(lambda_q)

        # do spatial derivation by fourier transforming twice
        n_psi = WaveFunction2D(self.paramObj)
        n_psi.setPsiHat((my_p + lambda_q) * self.psi_hat_array)
        n_psi.calcIFFT()

        self.nabla_psi_array = n_psi.psi_array
        self.nabla_psi_contains_values = True
        return self.nabla_psi_array

    @jit
    def calcL(self):
        """Calculates the angular momentum operator acting on the wavefunction.
        This is done by L Psi = (x*d/dy - y*d/dx) Psi [1].

        Returns:
            The Wavfunction after the angular momentum operator was applied.
        """
        # set some aliases
        a, b, c, d = self.paramObj.getBoundaries()
        M, N = self.paramObj.getResolution()

        # calculate the FFT of Psi
        if not self.psi_hat_contains_values:
            print("[WARNING] Calculating psi_hat since it is empty")
            self.calcFFT()

        # calculating D_x(Psi) and D_y(Psi) first to later Calculate L
        # set up grid in fourier space
        p = np.arange(-M//2, M//2, 1)
        q = np.arange(-N//2, N//2, 1)
        pp, qq = np.meshgrid(p, q, indexing='ij')

        # set up 'derivation' constants in fourier space
        lambda_q = 2*qq*np.pi/(d-c)
        my_p = 2*pp*np.pi/(b-a)

        # shift 'derivation' constants, will give wrong results otherwise
        my_p = np.fft.fftshift(my_p)
        lambda_q = np.fft.fftshift(lambda_q)

        # Calculate single spatial derivative in x direction
        Dx_psi = WaveFunction2D(self.paramObj)
        Dx_psi.setPsiHat(my_p * self.psi_hat_array)
        Dx_psi.calcIFFT()

        # Calculate single spatial derivative in y direction
        Dy_psi = WaveFunction2D(self.paramObj)
        Dy_psi.setPsiHat(lambda_q * self.psi_hat_array)
        Dy_psi.calcIFFT()

        #####
        # This code does the same thing but is slower. Easyer to read though.
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
        #####


        # adding Dx and Dy up to L
        xx, yy = np.meshgrid(self.paramObj.x, self.paramObj.y, sparse=False, indexing='ij')

        self.L_psi_array = xx * Dy_psi.psi_array - yy * Dx_psi.psi_array
        self.L_psi_contains_values = True
        return self.L_psi_array

    def calcG_m(self, psi_m, alpha):
        """Calculates G_m which contains the action of potential, the density and the rotation term of the GPE on the wavefunction,
        as well as a stabilization factor.
        Uses the formula give in [1].
        
        Arguments:
            psi_m: A wavefunction2D instance.
            alpha: A stabilization parameter.

        Returns:
            A 2D numpy array that contains G_m.
        """
        # check if all parameters have the right form
        if alpha < 0:
            print("[ALERT] Alpha < 0!")
        if type(psi_m) != WaveFunction2D:
            raise TypeError("Parameter Psi_m has to be of type WaveFunction2D.")
        if not self.psi_contains_values or not psi_m.psi_contains_values or not psi_m.L_psi_contains_values:
            raise ValueError("Something was not calculated...")
        
        # adding each term of G_m seperately to avoid mistakes.
        # using formula from [1]
        g = alpha * psi_m.psi_array + 0j
        g -= self.paramObj.V * psi_m.psi_array 
        g -= self.paramObj.beta2 * np.abs(self.psi_array)**2 * psi_m.psi_array 
        g += self.paramObj.omega * psi_m.L_psi_array
        return g


