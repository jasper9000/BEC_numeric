import numpy as np
from enum import IntEnum

class PotentialChoice(IntEnum):
    '''A simple enumerator to aviod confusion.
    '''
    HARMONIC = 0
    HARMONIC_QUARTIC = 1
    HARMONIC_OPTIC = 2
    NOT_IMPLEMENTED = -1

class Psi0Choice(IntEnum):
    '''A simple enumerator to aviod confusion.
    '''
    THOMAS_FERMI = 0
    GAUSS = 1
    NOT_IMPLEMENTED = -1


class ParameterObject:
    '''This class holds all parameters that are required to set up the simulation.
    It acts like a big wrapper object with some functionality.

    For a detailed description of all Parameters, read the README.md file in the root directory of this reposit.
    Attributes:
        x_high, x_low, y_high, y_low: Bounds for the grid.
        resolutionX, resolutionY: Gird resolution in x/y direction.
        beta2: self-interaction strength.
        omega: rotation frequency.
        epsilon_limit: exit condition.
        epsilon_thershold: a measure for the amout of frames saved.
        dt: time step.
        maxIterations: exit condition.
        filename: filename of the .hdf5 file.
        potential_choice: A PotentialChoice Enum.
        potential_parameters: A dictionary that contains all potential parameters.
        psi0_choice: A Psi0Choice Enum.
        psi0_parameters: A dictionary that contains all initial wavefunction parameters.

        V: A 2D numpy array that contains the trapping potential.
        x, y: 1D numpy arrays that contain a linear space corresponding to the bounds and resolutions.
    '''
    def __init__(self, resolutionX = 256, resolutionY = 256,
    x_low = -16, x_high = 16, y_low = -16, y_high = 16,
    beta2 = 1000, omega = 0.9,
    epsilon_limit=1e-10, epsilon_threshold=1, dt=0.005, maxIterations=30_000,
    filename='default.hdf5',
    potential_choice=PotentialChoice.HARMONIC, potential_parameters={'gamma_y':1, 'alpha':1.2, 'kappa_quartic':0.3, 'kappa_optic':0.7, 'V0':5},
    psi0_choice=Psi0Choice.THOMAS_FERMI, psi0_parameters={'gamma_y':1, 'sigma':1, 'x0':0, 'y0':0}):
        '''Initializes the instance with all the given parameters.
        Contains all standard parameters.
        '''

        self.resolutionX = resolutionX
        self.resolutionY = resolutionY

        # set the bounddaries of the image box
        self.x_low = x_low
        self.x_high = x_high
        self.y_low = y_low
        self.y_high = y_high

        # calculate the spatial step and make a 2D coordinate array
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
        self.potential_choice = potential_choice
        self.potential_parameters = potential_parameters
        self.V = None

        self.psi0_choice = psi0_choice
        self.psi0_parameters = psi0_parameters

    def __repr__(self):
        '''This function is called when a ParameterObject is converted to a string and returns all information as a formated string.
        '''
        a = ("x_low = {}, x_high = {}\n"
            "y_low = {}, y_high = {}\n"
            "res_x = {}, res_y = {}\n"
            "beta = {}, omega = {}\n"
            "epsilon = {}, epsilon_threshold = {}, dt = {}, maxIter = {}\n"
            "filename = {}\n\n"
            "Potential : {}\n"
            "Potential parameters : {}\n\n"
            "Psi0 : {}\n"
            "Psi0 parameters : {}")
        a = a.format(self.x_low, self.x_high, self.y_low, self.y_high, self.resolutionX, self.resolutionY,
        self.beta2, self.omega, self.epsilon_limit, self.epsilon_threshold, self.dt,
        self.maxIterations, self.filename, self.potential_choice, self.potential_parameters, self.psi0_choice, self.psi0_parameters)
        return a

    def updateGrid(self):
        '''This function updates the 2D grid array with the current values for the bound and resolution.
        '''
        self.dx = (self.x_high - self.x_low)/self.resolutionX
        self.x = np.linspace(self.x_low, self.x_high, self.resolutionX)

        self.dy = (self.y_high - self.y_low)/self.resolutionY
        self.y = np.linspace(self.y_low, self.y_high, self.resolutionY)

    def initVharmonic(self, V0 = 1, gamma_y = 1):
        '''This function initializes a harmonic potential.
        '''
        self.updateGrid()
        xx, yy = np.meshgrid(self.x, self.y, sparse=False, indexing='ij')
        self.V = V0 * 0.5*(xx**2 + (gamma_y*yy)**2)

    def initVharmonic_quartic(self, alpha=1.3, kappa=0.3):
        '''This function initializes a harmonic + quartic potential V(r) ~ r^2 + r^4.
        '''
        self.updateGrid()
        xx, yy = np.meshgrid(self.x, self.y, sparse=False, indexing='ij')
        self.V = (1-alpha)*(xx**2 + yy**2) + kappa * (xx**2+yy**2)**2
    
    def initVperiodic(self, V0 = 1, kappa = np.pi):
        '''This function initializes a harmonic + periodic (optic) potential V(r) ~ r^2 + sin^2(k*r).
        '''
        self.updateGrid()
        xx, yy = np.meshgrid(self.x, self.y, sparse=False, indexing='ij')
        Vopt = V0*(np.sin(kappa*xx)**2 + np.sin(kappa*yy)**2)
        self.V = 0.5 * (xx**2 + yy**2 + Vopt)

    def getResolution(self):
        '''Returns a 2 element tuple with the X- and Y- resolution
        '''
        return (self.resolutionX, self.resolutionY)

    def getBoundaries(self):
        '''Returns a 4 element tuple (x_low, x_high, y_low, y_high)
        '''
        return (self.x_low, self.x_high, self.y_low, self.y_high)

    def initV(self):
        '''Initializes the right potential based on potential_choice.
        '''
        self.updateGrid()
        if self.potential_choice == PotentialChoice.HARMONIC:
            self.initVharmonic(1, self.potential_parameters['gamma_y'])
        elif self.potential_choice == PotentialChoice.HARMONIC_QUARTIC:
            self.initVharmonic_quartic(self.potential_parameters['alpha'], self.potential_parameters['kappa_quartic'])
        elif self.potential_choice == PotentialChoice.HARMONIC_OPTIC:
            self.initVperiodic(self.potential_parameters['V0'], self.potential_parameters['kappa_optic'])
        else:
            raise ValueError("Potential choice not recognized.")
