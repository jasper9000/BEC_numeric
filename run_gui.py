import numpy as np
import tkinter as tk

from brain import ParameterApp, WaveFunction2D, ImaginaryTimeStepper
from time import time

if __name__ == "__main__":
    root = tk.Tk("Bose Einstein Kondensation")
    root.title("Numerical ground state of rotating Bose-Einstein Condensates : Parameter Selection")
    app = ParameterApp(root)
    root.mainloop()

    if app.pressedStart:
        start = time()
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
        print("[INFO] This calculation took {:.2} s.".format(time()-start))