import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from brain import ParameterObject, WaveFunction2D, ImaginaryTimeStepper, PotentialChoice, Psi0Choice


def plot2D(psi):
    plt.imshow(np.abs(psi))
    plt.colorbar()
    plt.show()

def display_psi_array(array, playback_speed=20, dynamic_colorbar=True):
    # a simple function that displays arrays of 2D image data as a animation
    fig, ax = plt.subplots()
    shape = array[0].shape
    data = np.random.rand(shape[0],shape[1])
    im = ax.imshow(data, cmap='jet', animated=True, vmin=0, vmax=1)
    fig.colorbar(im)
    tx = ax.set_title('Frame 0')

    def update(i):
        im.set_data(array[i])
        tx.set_text("Frame {}".format(i))
        if dynamic_colorbar:
            vmin = np.min(array[i])
            vmax = np.max(array[i])
            im.set_clim(vmin,vmax)
        return im
    anim = animation.FuncAnimation(fig, update, blit=False, frames=len(array), interval=playback_speed)
    plt.show()
    return anim


#### initialize parameters
res_x = 256
res_y = 256
x_low = -16
x_high = 16
y_low = -16
y_high = 16
beta2 = 1000
omega = 0.9,
epsilon_limit=1e-10
epsilon_threshold=1
dt=0.005
maxIterations=30_000
filename='default.hdf5'
potential_choice=PotentialChoice.HARMONIC
potential_parameters={'gamma_y':1, 'alpha':1.2, 'kappa_quartic':0.3, 'kappa_optic':0.7, 'V0':5}
psi0_choice=Psi0Choice.THOMAS_FERMI
psi0_parameters={'gamma_y':1, 'sigma':1, 'x0':0, 'y0':0}

#### initialize objects

p = ParameterObject(res_x, res_y, x_low, x_high, y_low, y_high,
                    beta2, omega, epsilon_limit, epsilon_threshold, dt, maxIterations,
                    filename, potential_choice, potential_parameters,
                    psi0_choice, psi0_parameters)
p.initV()

psi0 = WaveFunction2D(p)
psi0.initPsi_0()


i = ImaginaryTimeStepper(psi0, p)

# start the simulation
i.BFSP()

# i.dataM.displayFrames(30)

# it's better to close the file manually...
i.dataM.closeFile()

############################################
# THE OLD METHOD, BESP
# DOES NOT WORK VERY WELL...

# # calculate frames
# # just do 20 frames as an example
# frames = [i.returnFrame()]
# for t in range(20):
#     print("Calculating Frame {}".format(t+1))
#     i.calculate_time_step()
#     frames.append(i.returnFrame())
#     print("Epsilon_t = {}".format(np.max(np.abs(frames[t] - frames[t+1]))))

# # display the frames
# display_psi_array(frames, 200)
############################################
