import numpy as np
from wave_function import ParameterObject, WaveFunction2D
from numba import jit


class ImaginaryTimeStepper:
    def __init__(self, psi_0, parameterObject, epsilon_iteration_step_limit = 10e-5, dtInit = 0.005, maxTimeSteps = 500):
        if type(psi_0) != WaveFunction2D:
            raise TypeError("Parameter Psi_0 is not of type WaveFunction2D")
        self.psi_0 = psi_0

        if type(parameterObject) != ParameterObject:
            raise TypeError("Parameter parameterObject is not of type ParameterObject")
        self.paramObj = parameterObject

        self.dt = dtInit # the initial value for the first time step
        self.epsilon_iteration_step_limit = epsilon_iteration_step_limit # value for the accepted error for the next time step
        self.maxTimeSteps = maxTimeSteps # max number of allowed time steps

        # set up array for the nth time step, starting at 0
        self.psi_n = self.psi_0
        self.n = 0

    def returnFrame(self):
        # returns |psi|**2 from the current time
        return np.abs(self.psi_n.psi_array)**2

    def calcAlpha(self):
        # calculate alpha
        b_ = self.paramObj.V + self.paramObj.beta2*np.abs(self.psi_n.psi_array)**2
        bmin = np.min(b_)
        bmax = np.max(b_)
        alpha = 0.5 * (bmax + bmin)
        return alpha

    @jit
    def calcNextPsi_m(self, G_m, dt, alpha):
        if type(G_m) != WaveFunction2D:
            raise TypeError("Parameter G_m has to be of type WaveFunction2D.")

        a, b, c, d = self.paramObj.getBoundaries()
        M, N = self.paramObj.getResolution()

        psi_m_next = WaveFunction2D(self.paramObj)
        

        p = np.arange(-M//2, M//2, 1)
        q = np.arange(-N//2, N//2, 1)
        pp, qq = np.meshgrid(p, q, indexing='ij')

        lambda_q = 2*qq*np.pi/(d-c)
        my_p = 2*pp*np.pi/(b-a)

        my_p = np.fft.fftshift(my_p)
        lambda_q = np.fft.fftshift(lambda_q)

        psi_m_hat = self.psi_n.psi_hat_array + dt * G_m.psi_hat_array
        psi_m_hat *= 2 / (2 + dt*(2*alpha + my_p**2 + lambda_q**2))

        # psi_m_hat = np.zeros((M, N)) + (0+0j)
        # for p in range(-M//2, M//2, 1):
        #     for q in range(-N//2, N//2, 1):
        #         lambda_q = 2*q*np.pi/(d-c)
        #         my_p = 2*p*np.pi/(b-a)
        #         psi_m_hat[p,q] = self.psi_n.psi_hat_array[p,q] + dt * G_m.psi_hat_array[p,q]
        #         psi_m_hat[p,q] *= 2 / (2 + dt*(2*alpha + my_p**2 + lambda_q**2))

        psi_m_next.setPsiHat(psi_m_hat)
        return psi_m_hat

        # old code, should do the same thing but is way slower

        # next_psi_hat = np.zeros((M,N)) + (0+0j)
        # for p in range(-M//2, M//2):
        #     for q in range(-N//2, N//2):
        #         my_p = 2*p*np.pi/(b-a)
        #         lambda_q = 2*q*np.pi/(d-c)
        #         next_psi_hat[p,q] = self.psi_n.psi_hat_array[p,q] + dt * G_m.psi_hat_array[p,q]
        #         next_psi_hat[p,q] *= 2 / (2 + dt*(2*alpha + my_p**2 + lambda_q**2))
        
        # psi_m_next.setPsiHat(next_psi_hat)
        # return next_psi_hat

    def calculate_time_step(self):
        # set up epsilon
        epsilon_iteration_step = 1
        psi_max_old = np.zeros(self.paramObj.getResolution())
        psi_max = self.psi_0.psi_array
        m = 0

        # calculate alpha for this time step
        alpha = self.calcAlpha()

        # set up initial values
        self.psi_n.calcFFT()
        psi_m = self.psi_n
        psi_m.calcFFT()
        psi_m.calcL()

        G_m = WaveFunction2D(self.paramObj)
        G_m.setPsi(self.psi_n.calcG_m(psi_m, alpha))
        G_m.calcFFT()

        while epsilon_iteration_step > self.epsilon_iteration_step_limit:
            # do the iteration over m, which converges towards the next psi_n
            m += 1
            psi_m.setPsiHat(self.calcNextPsi_m(G_m, self.dt, alpha))
            psi_m.calcIFFT()
            psi_m.calcL()

            G_m.setPsi(self.psi_n.calcG_m(psi_m, alpha))
            G_m.calcFFT()

            # calculate epsilon
            psi_max_old = psi_max
            psi_max = psi_m.psi_array
            epsilon_iteration_step = np.max(np.abs(psi_max_old - psi_max)) / self.dt
            # print('\tEpsilon m = {}, iteration step = {}'.format(m, epsilon_iteration_step))

        # end of iteration, psi_m (hopefully) converged to psi_n+1
        # renormalize psi_m for the next time step
        print("Took {} iteration steps.".format(m))
        psi_m.norm()
        self.psi_n = psi_m
        self.n += 1

