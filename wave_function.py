import numpy as np
import numpy.fft as fft


class ParameterObject:
    def __init__(self, resolutionX = 256, resolutionY = 256, x_low = -10, x_high = 10, y_low = -10, y_high = 10, V0 = 1, gamma_y = 1):
        self.resolutionX = resolutionX
        self.resolutionY = resolutionY

        # set the bounddaries of the image box
        self.x_low = x_low
        self.x_high = x_high
        self.y_low = y_low
        self.y_high = y_high

        # calculate the spatial step and make a coordinate array
        self.dx = (self.x_high - self.x_low)/self.resolutionX
        self.x = np.arange(self.x_low, self.x_high, self.dx)

        self.dy = (self.y_high - self.y_low)/self.resolutionY
        self.y = np.arange(self.y_low, self.y_high, self.dy)

        # initialize the potential array V

        self.V = np.zeros((self.resolutionX, self.resolutionY))
        self.initVharmonic(V0, gamma_y)

    def initVharmonic(self, V0 = 0, gamma_y = 1):
        xx, yy = np.meshgrid(self.x, self.y, sparse=False, indexing='xy')
        self.V = V0 * 0.5*(xx**2 + (gamma_y*yy)**2)

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
        self.psi_array = np.zeros(self.paramObj.getResolution())

        self.psi_hat_contains_values = False
        self.psi_hat_array = np.zeros(self.paramObj.getResolution())

        self.L_psi_contains_values = False
        self.L_psi_array = np.zeros(self.paramObj.getResolution())
    
    def setPsi(self, array):
        if array.shape != self.paramObj.getResolution():
            raise ValueError("Shape {} of input array does not match the reesolution {}.".format(array.shape, self.paramObj.getResolution()))
        self.psi_array = array
        self.psi_contains_values = True

    def setPsiHat(self, array):
        if array.shape != self.paramObj.getResolution():
            raise ValueError("Shape {} of input array does not match the resolution {}.".format(array.shape, self.paramObj.getResolution()))
        self.psi_hat_array = array
        self.psi_hat_contains_values = True

    def initPsiGauss(self, sigma=1, x0=0, y0=0):
        xx, yy = np.meshgrid(self.paramObj.x, self.paramObj.y, sparse=False, indexing='xy')

        self.psi_array = 1/(sigma**2) * np.exp(-0.5*((xx-x0)**2 + (yy-y0)**2)/sigma**2)
        self.norm()
        self.psi_contains_values = True
        return self.psi_array

    def norm(self):
        self.psi_array /= np.sqrt( np.sum(np.abs(self.psi_array)**2) * self.paramObj.dx * self.paramObj.dy )
        return self.psi_array

    def calcFFT(self):
        if not self.psi_contains_values:
            raise ValueError("Psi does not contain values or Psi was not initialized!")
        else:
            self.psi_hat_array = np.fft.fft2(self.psi_array)/(np.prod(self.paramObj.getResolution()))
            self.psi_hat_contains_values = True
            return self.psi_hat_array

    def calcIFFT(self):
        if not self.psi_hat_contains_values:
            raise ValueError("Psi_hat does not contain values or Psi_hat was not initialized!")
        else:
            self.psi_array = np.fft.ifft2(self.psi_hat_array) * (np.prod(self.paramObj.getResolution()))
            self.psi_contains_values = True
            return self.psi_array

    def calcL(self):
        # set some aliases for resolution and boundaries to make the code more readable
        a, b, c, d = self.paramObj.getBoundaries()
        M, N = self.paramObj.getResolution()

        # calculate the FFT of Psi
        if not self.psi_hat_contains_values:
            print("[WARNING] calculating psi_hat since it is empty")
            self.calcFFT()
        else:
            print("[WARNING] psi_hat has values, will use them and NOT calculate again")

        # calculating D_x(Psi) and D_y(Psi) first to later Calculate L
        p = np.arange(-M//2, M//2, 1)
        q = np.arange(-N//2, N//2, 1)
        pp, qq = np.meshgrid(p, q, indexing='xy')

        lambda_q = 2*qq*np.pi/(d-c)
        my_p = 2*pp*np.pi/(b-a)

        Dx_psi = WaveFunction2D(self.paramObj)
        Dx_psi.setPsiHat(my_p * self.psi_hat_array)
        Dx_psi.calcIFFT()
        Dx_psi.psi_array /= np.prod( self.paramObj.getResolution() )

        Dy_psi = WaveFunction2D(self.paramObj)
        Dy_psi.setPsiHat(lambda_q * self.psi_hat_array)
        Dy_psi.calcIFFT()
        Dy_psi.psi_array /= np.prod( self.paramObj.getResolution() )

        # adding Dx and Dy up to L
        xx, yy = np.meshgrid(self.paramObj.x, self.paramObj.y, sparse=False, indexing='xy')

        self.L_psi = xx * Dy_psi.psi_array - yy * Dx_psi.psi_array
        return self.L_psi





p = ParameterObject(resolutionX=501, resolutionY=501)
w = WaveFunction2D(p)
w.initPsiGauss(sigma=0.5)

w.calcFFT()
#w.calcIFFT()
w.calcL()

import matplotlib.pyplot as plt

plt.imshow(np.abs(w.L_psi))
plt.show()
