import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from brain import *


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


#### initialize objects
gamma_y = 1.5
resolution = 256

p = ParameterObject(resolutionX=resolution, resolutionY=resolution, beta2=1000, omega=0.9)
p.initVharmonic(gamma_y=gamma_y)
# p.initVharmonic_quartic(1.2, 0.3)
# p.initVperiodic(V0=20, kappa=np.pi/4)

psi0 = WaveFunction2D(p)
# psi0.initPsi_0()
# psi0.initPsiGauss(sigma=2.5, x0=1.5, y0=0)
# psi0.initPsiGauss_double(sigma=1, x0=5, y0=0)
psi0.initThomasFermi(gamma_y=gamma_y)


i = ImaginaryTimeStepper(psi0, p, epsilon_iteration_step_limit=1e-15, dtInit=0.002, maxIterations=50_000, filename='D:/bec_data/gamma.hdf5')

# BFSP
i.BFSP(3)

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
