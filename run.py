import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from wave_function import ParameterObject, WaveFunction2D
from time_stepper import ImaginaryTimeStepper



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
            print(vmin, vmax)
            im.set_clim(vmin,vmax)
        return im
    anim = animation.FuncAnimation(fig, update, blit=False, frames=len(array), interval=playback_speed)
    plt.show()
    return anim


# initialize objects
p = ParameterObject(resolutionX=256, resolutionY=256, beta2=100, omega=0.8)
w = WaveFunction2D(p)
w.initPsi_0()
i = ImaginaryTimeStepper(w, p, epsilon_iteration_step_limit=10e-3)

# calculate frames
# just do 20 frames as an example
frames = [i.returnFrame()]
for t in range(20):
    print("Calculating Frame {}".format(t+1))
    i.calculate_time_step()
    frames.append(i.returnFrame())

# display the frames
display_psi_array(frames, 150)