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

from .wave_function import WaveFunction2D
from .parameter_object import ParameterObject
from .data_manager import DataManager

import numpy as np
from numba import jit
from time import time

def timer(func):
    '''This function can be used as a decorator to time the execution of another function.
    '''
    def wrapper(*args, **kwargs):
        start = time()
        r = func(*args, **kwargs)
        print("Function {} took {:.4} s".format(func.__name__, time() - start))
        return r
    return wrapper

class ImaginaryTimeStepper:
    '''This class is used to solve the Gross-Pitaevski equation (GPE).

    Attributes:
        psi_0: A WaveFunction2D instance that has the initial wavefunction in it.
        parameterObject: A ParameterObject instance that contains all setting and parameters for the simulation.
        dt: A float that is an alias for ParameterObject.dt
        epsilon_iteration_step_limit: A float that is an alias for ParameterObject.epsilon_limit
        maxIterations: A float that is an alias for ParameterObject.maxIterations
        psi_n: A WavveFunction2D instance that contains the current wave function at time step t_n
        n: An integer that refers to the current time step
        dataM: A DataManager instance that is handling the saving of the time steps to the disk.
        globalAttributes: A dictionary that has the contents of ParameterObject but the right format for the DataManager.
        '''
    def __init__(self, psi_0, parameterObject):
        '''Initializes an instance of this class.

        Arguments:
            psi_0: A WaveFunction2D instance that has the initial wavefunction in it.
            parameterObject: A ParameterObject instance that contains all setting and parameters for the simulation.
        '''
        # check if psi_0 is an instance of WaveFunction2D
        if type(psi_0) != WaveFunction2D:
            raise TypeError("Parameter Psi_0 is not of type WaveFunction2D")
        self.psi_0 = psi_0

        # check if parameterObject is an instance of ParameterObject
        if type(parameterObject) != ParameterObject:
            raise TypeError("Parameter parameterObject is not of type ParameterObject")
        self.paramObj = parameterObject

        # set up aliases for parameterObject
        self.dt = self.paramObj.dt
        self.epsilon_iteration_step_limit = self.paramObj.epsilon_limit
        self.maxIterations = self.paramObj.maxIterations

        # set up array for the nth time step, starting at 0
        self.psi_n = self.psi_0
        self.n = 0

        # set up data manager
        self.dataM = DataManager(self.paramObj.filename)
        self.dataM.newFile()
        self.globalAttributes = {
            'omega': self.paramObj.omega,
            'beta': self.paramObj.beta2,
            'dt': self.dt,
            'resX': self.paramObj.resolutionX,
            'resY': self.paramObj.resolutionY,
            'x_low' : self.paramObj.x_low,
            'x_high' : self.paramObj.x_high,
            'y_low' : self.paramObj.y_low,
            'y_high' : self.paramObj.y_high,
            'epsilon_limit': self.epsilon_iteration_step_limit,
            'epsilon_threshold' : self.paramObj.epsilon_threshold,
            'maxIterations': self.maxIterations,
            'potential_choice': int(self.paramObj.potential_choice),
            'psi0_choice' : int(self.paramObj.psi0_choice),
            'potential_gamma_y' : self.paramObj.potential_parameters['gamma_y'],
            'potential_alpha' : self.paramObj.potential_parameters['alpha'],
            'potential_kappa_quartic' : self.paramObj.potential_parameters['kappa_quartic'],
            'potential_kappa_optic' : self.paramObj.potential_parameters['kappa_optic'],
            'potential_V0' : self.paramObj.potential_parameters['V0'],
            'psi0_gamma_y' : self.paramObj.psi0_parameters['gamma_y'],
            'psi0_sigma' : self.paramObj.psi0_parameters['sigma'],
            'psi0_x0' : self.paramObj.psi0_parameters['x0'],
            'psi0_y0' : self.paramObj.psi0_parameters['y0'],
            'calculated_observables' : False
        }
        self.dataM.setGlobalAttributes(self.globalAttributes)
    
    def __del__(self):
        '''Destructor of this class, is called when the instance is deleted and closes any open files.
        '''
        self.dataM.closeFile()

    def returnFrame(self):
        '''Returns |psi|**2 of the current time step.
        '''
        return np.abs(self.psi_n.psi_array)**2

    def returnPsi(self):
        '''Returns psi of the current time step.
        '''
        return self.psi_n.psi_array

    def calcAlpha(self):
        '''Calculate stability parameter alpha.

        Returns:
            A (real) float.
        '''
        # This follows [1] and [2]. A stabilization parameter is used for faster convergence.
        b_ = self.paramObj.V + self.paramObj.beta2*np.abs(self.psi_n.psi_array)**2
        bmin = np.min(b_)
        bmax = np.max(b_)
        alpha = 0.5 * (bmax + bmin)
        # check whether the time step dt obeys the contraint.
        if self.dt > 2/(bmax+bmin):
            print("[WARNING] Delta t = {} is larger than time step constraint {}!".format(self.dt, 2/(bmax+bmin)))
        return alpha

    @jit
    def calcNextPsi_m(self, G_m, alpha):
        '''This function is used to iteratively solve the equation system in every time step.

        Arguments:
            G_m: A Wavefunction2D instance that contains all terms of the GPE that are not the second order spacial derivative. Calculated by WaveFunction2D.clacG_m.
            alpha: A float that is the stabilization parameter used in [1] and [2].
        '''
        # check if G_m is instance of WaveFunction2D
        if type(G_m) != WaveFunction2D:
            raise TypeError("Parameter G_m has to be of type WaveFunction2D.")
        
        # check if all required components of the wavefunctions have been calculated
        if not self.psi_n.psi_hat_contains_values or not G_m.psi_hat_contains_values:
            raise ValueError("Something was not calculated...")

        # set up aliases for grid parameters
        a, b, c, d = self.paramObj.getBoundaries()
        M, N = self.paramObj.getResolution()

        # set up result WaveFunction2D
        psi_m_next = WaveFunction2D(self.paramObj)
        
        # set up a grid in fourier space
        p = np.arange(-M//2, M//2, 1)
        q = np.arange(-N//2, N//2, 1)
        pp, qq = np.meshgrid(p, q, indexing='ij')

        # set up 'derivation' constants in fourier space
        lambda_q = 2*qq*np.pi/(d-c)
        my_p = 2*pp*np.pi/(b-a)

        # shift 'derivation' constants, will give wrong results otherwise
        my_p = np.fft.fftshift(my_p)
        lambda_q = np.fft.fftshift(lambda_q)

        # calculate next iteration step with scheme given in [1]
        psi_m_hat = self.psi_n.psi_hat_array + self.dt * G_m.psi_hat_array
        psi_m_hat *= 2 / (2 + self.dt*(2*alpha + my_p**2 + lambda_q**2))

        # the old version
        # does the same thing but way slower, CONFIRMED

        # psi_m_hat = np.zeros((M, N)) + (0+0j)
        # for p in range(-M//2, M//2, 1):
        #     for q in range(-N//2, N//2, 1):
        #         lambda_q = 2*q*np.pi/(d-c)
        #         my_p = 2*p*np.pi/(b-a)
        #         psi_m_hat[p,q] = self.psi_n.psi_hat_array[p,q] + dt * G_m.psi_hat_array[p,q]
        #         psi_m_hat[p,q] *= 2 / (2 + dt*(2*alpha + my_p**2 + lambda_q**2))

        # return the result
        psi_m_next.setPsiHat(psi_m_hat)
        return psi_m_hat
        
    @jit
    def calculate_time_step(self):
        '''This function is not used anymore. It was used in the Backwards Euler Pseudo Fourier Scheme with an iterative approach.
        It calculates the next time step t_n+1 from the time step t_n with an iterative scheme.
        '''
        # OLD
        # calculates psi for t_n+1 in BEFP
        # development was started with this approach, but since in the end a different approach was used, this function was not optimised.
        #####
        # set up epsilon (difference inbetween steps)
        epsilon_iteration_step = 1
        psi_max_old = np.zeros(self.paramObj.getResolution())
        psi_max = self.psi_0.psi_array
        m = 0

        # calculate alpha for this time step
        alpha = self.calcAlpha()

        # set up initial values for psi_n/m and calculate components
        self.psi_n.calcFFT()
        psi_m = self.psi_n
        psi_m.calcFFT()
        psi_m.calcL()

        G_m = WaveFunction2D(self.paramObj)
        G_m.setPsi(self.psi_n.calcG_m(psi_m, alpha))
        G_m.calcFFT()

        # start the iteration over m, which converges towards the next psi_n
        while epsilon_iteration_step > self.epsilon_iteration_step_limit:
            m += 1
            # calculate the next iterative solution. Scheme in [1].
            psi_m.setPsiHat(self.calcNextPsi_m(G_m, alpha))
            
            # calculate components for next iteration
            psi_m.calcIFFT()
            psi_m.calcL()

            G_m.setPsi(self.psi_n.calcG_m(psi_m, alpha))
            G_m.calcFFT()

            # calculate epsilon
            psi_max_old = psi_max
            psi_max = psi_m.psi_array
            epsilon_iteration_step = np.max(np.abs(psi_max_old - psi_max)) / self.dt
            # print('\tm = {}, Epsilon iteration step = {}, norm = {}'.format(m, epsilon_iteration_step, psi_m.getNorm()))

        # end of iteration, psi_m (hopefully) converged to psi_n+1
        # renormalize psi_m for the next time step
        print("Took {} iteration steps.".format(m))
        psi_m.norm()
        self.psi_n = psi_m
        self.n += 1

    def BFFP(self):
        '''This function solves the GPE in imaginary time and calculates the ground state of the system.

        A Backwards/Forwards Fourier Pseudo-spectral scheme is used to approximatively solve the GPE.
        The spatial derivatives are solved by Fourier Transformation while the time derivative is solved by the semi-implicit (symplectic) Euler scheme.
        For every time step, the equation system that arises is only solved approximately which imposes a constraint on the time step dt.
        This saves computational effort but limits the time step to small values. [2] 
        '''
        # set up epsilon
        epsilon_iteration_step = 1
        epsilon_sum = 0
        psi_max_old = np.zeros(self.paramObj.getResolution())
        psi_max = self.psi_0.psi_array
        self.n = 0

        G_m = WaveFunction2D(self.paramObj)
        

        # loop over imaginary time steps
        # conditions for loop are exit conditions
        while epsilon_iteration_step > self.epsilon_iteration_step_limit and self.n < self.maxIterations:
            # calculate alpha for this time step
            alpha = self.calcAlpha()

            # the linear system is aproximatively solved by only doing one iteration step to the iterative solution of the system.
            # calculate psi_hat, L_psi and G_n
            self.psi_n.calcFFT()
            self.psi_n.calcL()
            G_m.setPsi(self.psi_n.calcG_m(self.psi_n, alpha))
            G_m.calcFFT()

            # calculate the next psi_n
            self.n += 1
            self.psi_n.setPsiHat(self.calcNextPsi_m(G_m, alpha))
            self.psi_n.calcIFFT()
            self.psi_n.norm()

            # calculate epsilon
            psi_max_old = psi_max
            psi_max = self.psi_n.psi_array
            epsilon_iteration_step = np.max(np.abs(psi_max_old - psi_max)) / self.dt
            print('n = {}, Epsilon = {:1.3e}, Epsilon sum = {:1.2f}'.format(self.n, epsilon_iteration_step, epsilon_sum))
            epsilon_sum += epsilon_iteration_step

            # see if a frame has to be saved
            if epsilon_sum > self.paramObj.epsilon_threshold:
                attributes = {
                    'n': self.n,
                    't': self.n*self.dt,
                    'epsilon': epsilon_iteration_step
                }
                self.dataM.addDset(self.returnPsi(), attributes)
                epsilon_sum = 0
                print("Saved a frame. Frame number {}".format(self.dataM.incKey))
        
        # end of iteration, psi_n should (hopefully) be the ground state
        # add the last frame to the data manager
        attributes = {
                    'n': self.n,
                    't': self.n*self.dt,
                    'epsilon': epsilon_iteration_step
                }
        self.dataM.addDset(self.returnPsi(), attributes)
        self.dataM.file.attrs["n_frames"] = self.dataM.getNFrames()

        print("Took {} (time) iteration steps. Saved {} frames.".format(self.n, self.dataM.incKey))
