import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import tkinter as tk

from brain import ParameterApp, WaveFunction2D, ImaginaryTimeStepper


if __name__ == "__main__":
    root = tk.Tk("Bose Einstein Kondensation")
    app = ParameterApp(root)
    root.mainloop()

    if app.pressedStart:
        p = app.paramObj

        del app
        del root

        psi0 = WaveFunction2D(p)
        psi0.initPsi_0()

        i = ImaginaryTimeStepper(psi0, p)

        # BFSP
        i.BFSP()
        # it's better to close the file manually...
        i.dataM.closeFile()