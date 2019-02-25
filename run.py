import tkinter as tk
from tkinter import font

import numpy as np
from brain import ResultsApp, ParameterApp, WaveFunction2D, ImaginaryTimeStepper
from time import time
from enum import Enum

class ProgramChoice(Enum):
    '''This enumerator is just a good practice, but really not neccessary.
    It keeps track of which program the user has selected.
    '''
    PARAMETERS = 0 
    RESULTS = 1
    NOT_IMPLEMENTED = 2

class BECApp(tk.Frame):
    '''Class that produces the initial GUI where the user selects, which operation they want to perform.
    '''
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        
        # set up padding of GUI elements to make it look nicer.
        self.padx = 25
        self.pady = 25
        self.ipadx = 15
        self.ipady = 15

        self.init_frame()

    def init_frame(self):
        '''Initializes the GUI elements.
        '''
        self.top_level_frame = tk.Frame(self.parent)
        self.top_level_frame.pack()

        # set up fonts
        self.font_medium = font.Font(family="Helvetica", size=12, weight=tk.NORMAL)
        self.font_large = font.Font(family="Helvetica", size=18, weight=tk.NORMAL)

        t = ("Welcome!\n"
            "If you want to simulate a Bose-Einstein Condensate and choose the Parameters for the simulation,\n"
            "click the button on the left.\n"
            "If you have a finished simulation and want to look at the results,\n"
            "click the button in the right.")

        # set up welcome text
        info_label = tk.Label(self.top_level_frame, text=t, font=self.font_medium)
        info_label.pack(side=tk.TOP, padx=self.padx, pady=self.pady, ipadx=self.ipadx, ipady=self.ipady)
        
        # set up buttons
        self.parameters_button = tk.Button(self.top_level_frame, command=self.run_parameters, text="Parameters", font=self.font_large, borderwidth=5)
        self.parameters_button.pack(side=tk.LEFT, padx=self.padx, pady=self.pady, ipadx=self.ipadx, ipady=self.ipady)

        self.results_button = tk.Button(self.top_level_frame, command=self.run_results, text="Results", font=self.font_large, borderwidth=5)
        self.results_button.pack(side=tk.RIGHT, padx=self.padx, pady=self.pady, ipadx=self.ipadx, ipady=self.ipady)

    def run_parameters(self):
        '''Executed when the user chose the parameter app.
        Destroys window and spawns the parameter app.
        '''
        global root, choice
        choice = ProgramChoice.PARAMETERS
        root.destroy()

    def run_results(self):
        '''Executed when the user chose the results app.
        Destroys window and spawns the results app.
        '''
        global root, choice
        choice = ProgramChoice.RESULTS
        root.destroy()



if __name__ == "__main__":
    choice = None

    # set up the window for the startup GUI
    root = tk.Tk("Bose Einstein Kondensation")
    root.title("Numerical ground state of rotating Bose-Einstein Condensates")
    root.geometry('+30+30')
    app = BECApp(root)
    root.mainloop()

    # if case: react to user input
    if choice == ProgramChoice.PARAMETERS:
        # launch parameters app
        root = tk.Tk("Bose Einstein Kondensation")
        root.title("Numerical ground state of rotating Bose-Einstein Condensates : Parameter Selection")
        root.geometry("+40+40")
        app = ParameterApp(root)
        root.mainloop()

        # executed when the start simulation button was clicked
        if app.pressedStart:
            start = time()
            # set up parameters and initial wave function according to user input
            p = app.paramObj

            del app
            del root

            psi0 = WaveFunction2D(p)
            psi0.initPsi_0()

            i = ImaginaryTimeStepper(psi0, p)

            # starting the simulation
            i.BFSP()
            # it's better to close the file manually...
            i.dataM.closeFile()
            # time the simulation took
            print("[INFO] This calculation took {} s.".format(time()-start))

    elif choice == ProgramChoice.RESULTS:
        # Launch results app
        root = tk.Tk()
        root.geometry("+40+40")
        root.title("Numerical ground state of rotating Bose-Einstein Condensates : Result Presentation")
        app = ResultsApp(root)
        root.mainloop()
    else:
        # nothing was selected, quitting the program
        print("[INFO] Nothing was selected, quitting.")