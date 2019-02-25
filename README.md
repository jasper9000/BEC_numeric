# BEC_numeric

## Introduction
This repository contains the implementation of numerical methods to compute the ground state of a ideal Bose-Einstein condensate (BEC). The evolution of a BEC is governed by the Gross-Pitaevski-Equation (GPE). This python program therefore solves the GPE in imaginary time with the method of gradient flow to obtain the energatically lowest state of the BEC, which is the ground state.
The method is described by Zeng & Zhang in their [publication](https://doi.org/10.1016/j.cpc.2008.12.003) 'Efficiently computing vortex lattices in rapid rotating Bose–Einstein condensates'. This project mostly follws their methods.

## Requirements
This program was tested with Python 3.6, but other Python 3 versions may be supported.
It requires the following (free) libraries:
 - numpy
 - numba
 - matplotlib
 - pyh5 (to handle .hdf5 files)

All of these external libraries are contained within the Anaconda framework.

## Usage
To use this program, one just has to clone (copy) this repository to some location on their machine and execute the run.py file without any arguments.
```
python run.py
```
In the GUI the user can select whether they want to set up parameters for a calculation or review results of previous calcultaions.

In the following paragraph all the different parameters for the calculation are described.

## Parameter Description
### Grid parameters
 - x low
 - x high
 - y low
 - y high

These parameters control the simulated area in dimensionless units of the harmonic oscillator length. A square box is recommended. Default value: -16 for low, 16 for high.

 - resolution x
 - resolution y

These parameters control the number of simulated pixels in x/y direction. A low value will sometimes result in wrong simulations, but often gives a good approximation of the result and drastically reduces computing time. For an accurate result, a value larger than 256 is recommended. In the special case of a harmonic + quartic trap, a value larger than 512 is recommended. Also, equal resolution for both axes are recommended.

### Physical Parameters
 - Omega

This parameter is the (dimensionless) rotation frequency of the BEC. A value of 0 corresponds to no rotation. A value larger or equal to 1 will only give a result if the harmonic plus quartic potential and a very high resolution is selected. This is due to the fact that i.e. a harmonic potential is not 'steep' enough to contain the BEC while strong centrifugal forces act on it.

 - Beta

This parameter is a measure of the strength of the (repulsive) self-interaction of the BEC. A low repulsion corresponds to beta = 100 while a strong repulsion corresponds to beta = 10000.

**Initial Wavefunction**
- Thomas-Fermi approximation

This sets the initial wavefunction to the [Thomas-Fermi approximation](https://en.wikipedia.org/wiki/Gross%E2%80%93Pitaevskii_equation#Thomas%E2%80%93Fermi_approximation).
The parameter gamma controls whether the function is anisotropic in the y direction and streches/contracts it accordingly.

- Gauss

This sets the initial wave to a simple 2D Gauss curve.
Sigma controls the width of the curve while x0/y0 control the position of the maximum of the curve.

**Potential V**

- Harmonic

This sets the trapping potential to a simple 2D harmonic oscillator. The parameter gamma controls anisotropy in y direction.

- Harmonic + Quartic

This sets the trapping potential to a harmonic oscillator with an added term of fourth order. It produces a well that is width but much stepper on the sides compared to the unperturbed harmonic oscillator. It follows the formula V(r) = (1 - alpha) * r^2 + kappa * r^4

- Harmonic + Optic

This sets the trapping potential to a harmonic oscillator with an added oscillating term, which simulates an optical lattice. It follows the formula V(r) = 0.5 * r^2 + V0 * sin^2(kappa * r)

### Numerical Parameters
 - Delta t

This sets the numerical time step of the imaginary time evolution. With the method used there is a constraint for delta t which limits it to small time step, but this constraint is already integrated in the GUI.

 - Epsilon Limit Exponent

This sets one of the exit conditions for the simulation. Whenever the difference (epsilon) of the wavefunctions of two time steps is smaller than 10 to the power of the given exponent, the simulation is considered finished and the ground state is found. The smaller epsilon is inbetween two time steps the less changed in that time step. A smaller exponent will result in a longer, but more accurate simulation.

 - max. Iteration Steps

This is the second exit condition for the simulation. Whenever a maximum number of time steps have been simulated, the simulation stops..

 - Filename

This is the name of the .hdf5 file in which the wavefunctions of selected time steps of the simulation will be saved.

 - Epsilon Threshold
 
This parameter controls the amount of time steps (frames) that are saved to the .hdf5 file. A small value will result in many saved frames and a larger filesize while a larger number will result in less saved frames.

# TO DO
 - [ ] Remove library dependencies on mpl_toolkits and scipy (wave_function.py)
 - [ ] Add a small GUI where the user can select if they want to look at results or calculate something new
 - [ ] look at the time evolution without imaginary time, i.e. the real time evolution compared to finding the ground state
 - [ ] find eingenstates of the condensate for higher energies than the ground state
 - [ ] (maybe) look at and implement other ways to solve the GPE (split step fourier method for time derivative)