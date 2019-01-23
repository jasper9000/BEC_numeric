import numpy as np
from enum import Enum

class PotentialChoice(Enum):
    HARMONIC = 0
    HARMONIC_QUARTIC = 1
    HARMONIC_OPTIC = 2
    NOT_IMPLEMENTED = -1

class Psi0Choice(Enum):
    THOMAS_FERMI = 0
    GAUSS = 1
    NOT_IMPLEMENTED = -1


class ParameterObject:
    def __init__(self, resolutionX = 256, resolutionY = 256,
    x_low = -16, x_high = 16, y_low = -16, y_high = 16,
    beta2 = 1000, omega = 0.9,
    epsilon_limit=1e-10, epsilon_threshold=1, dt=0.005, maxIterations=30_000,
    filename='default.hdf5',
    potential_choice=PotentialChoice.HARMONIC, potential_parameters={'gamma_y':1, 'alpha':1.2, 'kappa_quartic':0.3, 'kappa_optic':0.7, 'V0':5},
    psi0_choice=Psi0Choice.THOMAS_FERMI, psi0_parameters={'gamma_y':1, 'sigma':1, 'x0':0, 'y0':0}):

        self.resolutionX = resolutionX
        self.resolutionY = resolutionY

        # set the bounddaries of the image box
        self.x_low = x_low
        self.x_high = x_high
        self.y_low = y_low
        self.y_high = y_high

        # calculate the spatial step and make a coordinate array
        self.updateGrid()

        # constants for the BEC itself
        self.beta2 = beta2
        self.omega = omega

        # numerical parameters
        self.epsilon_limit = epsilon_limit
        self.epsilon_threshold = epsilon_threshold
        self.dt = dt
        self.maxIterations = maxIterations
        self.filename = filename

        # potential choice and psi_0 choice
        self.choice_V = potential_choice
        self.choice_V_parameters = potential_parameters
        self.V = None

        self.choice_psi0 = psi0_choice
        self.choice_psi0_parameters = psi0_parameters

    def __repr__(self):
        a = "x_low = {}, x_high = {}\n\
y_low = {}, y_high = {}\n\
res_x = {}, res_y = {}\n\
beta = {}, omega = {}\n\
epsilon = {}, epsilon_threshold = {}, dt = {}, maxIter = {}\n\
filename = {}\n\n\
Potential : {}\n\
Potential parameters : {}\n\n\
Psi0 : {}\n\
Psi0 parameters : {}".format(self.x_low, self.x_high, self.y_low, self.y_high, self.resolutionX, self.resolutionY,
        self.beta2, self.omega, self.epsilon_limit, self.epsilon_threshold, self.dt,
        self.maxIterations, self.filename, self.choice_V, self.choice_V_parameters, self.choice_psi0, self.choice_psi0_parameters)
        return a

    def updateGrid(self):
        self.dx = (self.x_high - self.x_low)/self.resolutionX
        self.x = np.linspace(self.x_low, self.x_high, self.resolutionX)

        self.dy = (self.y_high - self.y_low)/self.resolutionY
        self.y = np.linspace(self.y_low, self.y_high, self.resolutionY)

    def initVharmonic(self, V0 = 1, gamma_y = 1):
        self.updateGrid()
        xx, yy = np.meshgrid(self.x, self.y, sparse=False, indexing='ij')
        self.V = V0 * 0.5*(xx**2 + (gamma_y*yy)**2)

    def initVharmonic_quartic(self, alpha=1.3, kappa=0.3):
        self.updateGrid()
        xx, yy = np.meshgrid(self.x, self.y, sparse=False, indexing='ij')
        self.V = (1-alpha)*(xx**2 + yy**2) + kappa * (xx**2+yy**2)**2
    
    def initVperiodic(self, V0 = 1, kappa = np.pi):
        self.updateGrid()
        xx, yy = np.meshgrid(self.x, self.y, sparse=False, indexing='ij')
        Vopt = V0*(np.sin(kappa*xx)**2 + np.sin(kappa*yy)**2)
        self.V = 0.5 * (xx**2 + yy**2 + Vopt)

    def getResolution(self):
        # returns a 2 element tuple with the X- and Y- resolution
        return (self.resolutionX, self.resolutionY)

    def res(self):
        # alias for self.getResolution()  ...  it's just shorter
        return self.getResolution()

    def getBoundaries(self):
        # returns a 4 tuple with the x- and y-boundaries
        return (self.x_low, self.x_high, self.y_low, self.y_high)

    def initV(self):
        self.updateGrid()
        if self.choice_V == PotentialChoice.HARMONIC:
            self.initVharmonic(1, self.choice_V_parameters['gamma_y'])
        elif self.choice_V == PotentialChoice.HARMONIC_QUARTIC:
            self.initVharmonic_quartic(self.choice_V_parameters['alpha'], self.choice_V_parameters['kappa_quartic'])
        elif self.choice_V == PotentialChoice.HARMONIC_OPTIC:
            self.initVperiodic(self.choice_V_parameters['V0'], self.choice_V_parameters['kappa_optic'])
        else:
            raise ValueError("Potential choice not recognized.")
