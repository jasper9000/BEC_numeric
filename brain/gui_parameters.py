import tkinter as tk
import tkinter.font as font
from tkinter import messagebox
from tkinter.filedialog import asksaveasfilename

import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import numpy as np
from .parameter_object import ParameterObject, PotentialChoice, Psi0Choice
from .wave_function import WaveFunction2D

class ParameterApp(tk.Frame):
    '''This class produces the GUI in which the user can select the parameters for a simulation.
    '''
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # set up standard parameters
        self.paramObj = ParameterObject()
        self.paramObj.initV()

        # set up dt constraint
        self.dt_constraint = None
        self.calc_dt_constraint()

        self.pressedStart = False

        self.padx = 5
        self.pady = 5
        
        # initializes the GUI
        self.init_frames()
        self.init_menu()
        self.init_top_left()
        self.init_top_right()
        self.init_bottom_left()
        self.init_bottom_right()

    def init_menu(self):
        # set up the drop down menu
        self.root_menu = tk.Menu(self.parent)
        self.parent.config(menu=self.root_menu)
        self.preset_menu = tk.Menu(self.root_menu)
        self.root_menu.add_cascade(label="Example Parameter Presets", menu=self.preset_menu)
        self.preset_menu.add_command(label="No rotation (quick, a few minutes)", command=self.loadPresetNoRotation)
        self.preset_menu.add_command(label="Harmonic trap with rotating weakly self-interacting BEC (15-25 min)", command=self.loadPresetBetaSmall)
        self.preset_menu.add_command(label="Harmonic trap with rotating strongly self-interacting BEC (45 min)", command=self.loadPresetBetaLarge)
        self.preset_menu.add_command(label="Anisotropic harmonic trap", command=self.loadPresetAnisotropic)
        self.preset_menu.add_command(label="Harmonic + optic trap", command=self.loadPresetHarmonicOptic)
        self.preset_menu.add_command(label="Harmonic + quartic trap (this takes very long, ca. 24h)", command=self.loadPresetHarmonicQuartic)

    def init_frames(self):
        # initializes the top level frame which is the highest frame in the hierachie
        self.topLevelFrame = tk.Frame(self.parent)
        self.topLevelFrame.grid()

        # set up 4 frames for all the corners
        self.top_left = tk.LabelFrame(self.topLevelFrame, text='Grid Parameters', labelanchor='nw', width=400, height=400, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.top_right = tk.LabelFrame(self.topLevelFrame, text='Physical Parameters', labelanchor='nw', width=400, height=400, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.bottom_left = tk.LabelFrame(self.topLevelFrame, text='Numerical Parameters', labelanchor='nw', width=400, height=400, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.bottom_right = tk.LabelFrame(self.topLevelFrame, text='Potential V', labelanchor='nw', width=400, height=400, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        

        # layout all of the main containers
        # self.parent.grid_rowconfigure(1, weight=1)
        # self.parent.grid_columnconfigure(0, weight=1)

        # place frames on the window
        self.top_left.grid(row=0, column=0, padx=self.padx, pady=self.pady, sticky='')
        self.top_right.grid(row=0, column=1, padx=self.padx, pady=self.pady, sticky='')
        self.bottom_left.grid(row=1, column=0, padx=self.padx, pady=self.pady, sticky='')
        self.bottom_right.grid(row=1, column=1, padx=self.padx, pady=self.pady, sticky='')

    def init_top_left(self):
        # initializes the top left frame which contains the grid parameters
        # set up sliders for each parameter

        # x low
        x_low_label = tk.Label(self.top_left, text='x low')
        self.x_low_sv = tk.IntVar(value=self.paramObj.x_low)
        self.x_low_sv.trace_add("write", self.onChange)
        self.x_low_scale = tk.Scale(self.top_left, from_=-10, to_=-25, resolution=1, tickinterval=5, orient=tk.HORIZONTAL, variable=self.x_low_sv, length=150)

        # x high
        x_high_label = tk.Label(self.top_left, text='x high')
        self.x_high_sv = tk.IntVar(value=self.paramObj.x_high)
        self.x_high_sv.trace_add("write", self.onChange)
        self.x_high_scale = tk.Scale(self.top_left, from_=10, to_=25, resolution=1, tickinterval=5, orient=tk.HORIZONTAL, variable=self.x_high_sv, length=150)

        # y low
        y_low_label = tk.Label(self.top_left, text='y low')
        self.y_low_sv = tk.IntVar(value=self.paramObj.y_low)
        self.y_low_sv.trace_add("write", self.onChange)
        self.y_low_scale = tk.Scale(self.top_left, from_=-10, to_=-25, resolution=1, tickinterval=5, orient=tk.HORIZONTAL, variable=self.y_low_sv, length=150)

        # y high
        y_high_label = tk.Label(self.top_left, text='y high')
        self.y_high_sv = tk.IntVar(value=self.paramObj.y_high)
        self.y_high_sv.trace_add("write", self.onChange)
        self.y_high_scale = tk.Scale(self.top_left, from_=10, to_=25, resolution=1, tickinterval=5, orient=tk.HORIZONTAL, variable=self.y_high_sv, length=150)

        # place all the sliders on the grid
        x_low_label.grid(row=0, column=0)
        self.x_low_scale.grid(row=0, column=1)

        x_high_label.grid(row=0, column=2)
        self.x_high_scale.grid(row=0, column=3)

        y_low_label.grid(row=1, column=0)
        self.y_low_scale.grid(row=1, column=1)

        y_high_label.grid(row=1, column=2)
        self.y_high_scale.grid(row=1, column=3)

        # set up sliders for resolution parameters
        color_red = '#FF9A7E'
        self.resolution_frame = tk.Frame(self.top_left, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3, bg=color_red)
        self.resolution_frame.grid(row=2, column=0, columnspan=5, padx=self.padx, pady=self.pady)

        res_x_label = tk.Label(self.resolution_frame, text='Resolution in x-direction', bg=color_red)
        self.res_x_sv = tk.IntVar(value=self.paramObj.resolutionX)
        self.res_x_sv.trace_add("write", self.onChange)
        self.res_x_entry = tk.Scale(self.resolution_frame, from_=16, to_=1024, resolution=32, tickinterval=128, orient=tk.HORIZONTAL, variable=self.res_x_sv, length=200, bg=color_red, highlightthickness=0)

        res_y_label = tk.Label(self.resolution_frame, text='Resolution in y-direction', bg=color_red)
        self.res_y_sv = tk.IntVar(value=self.paramObj.resolutionY)
        self.res_y_sv.trace_add("write", self.onChange)
        self.res_y_entry = tk.Scale(self.resolution_frame, from_=16, to_=1024, resolution=32, tickinterval=128, orient=tk.HORIZONTAL, variable=self.res_y_sv, length=200, bg=color_red, highlightthickness=0)

        # place resolution sliders on the window
        res_x_label.grid(row=0, column=0)
        self.res_x_entry.grid(row=0, column=1)

        res_y_label.grid(row=1, column=0)
        self.res_y_entry.grid(row=1, column=1)

    def init_top_right(self):
        # initializes the top right frame which contains the physical parameters
        # sliders for omega and beta
        omega_label = tk.Label(self.top_right, text='Omega')
        self.omega_sv = tk.DoubleVar(value=self.paramObj.omega)
        self.omega_sv.trace_add("write", self.onChange)
        self.omega_scale = tk.Scale(self.top_right, from_=0, to_=3, resolution=0.05, tickinterval=1, orient=tk.HORIZONTAL, variable=self.omega_sv, length=150)

        beta_label = tk.Label(self.top_right, text='Beta')
        self.beta_sv = tk.DoubleVar(value=self.paramObj.beta2)
        self.beta_sv.trace_add("write", self.onChange)
        self.beta_scale = tk.Scale(self.top_right, from_=100, to_=10000, resolution=100, tickinterval=4000, orient=tk.HORIZONTAL, variable=self.beta_sv, length=150)

        omega_warning_label = tk.Label(self.top_right, text="The Parameter Omega is the dimensionless rotation frequency of the BEC.\nA value higher than 1 will only produce a result if\nthe harmonic + quartic potential is selected.")

        omega_label.grid(row=0, column=0)
        self.omega_scale.grid(row=0, column=1)

        beta_label.grid(row=0, column=3)
        self.beta_scale.grid(row=0, column=4)

        omega_warning_label.grid(row=1, column=0, columnspan=6)

        # dropdown menu for the initial wave function Psi 0
        self.psi0_option_list = ('Thomas-Fermi-Approximation', 'Gauss')
        self.psi0_option_sv = tk.StringVar(value=self.psi0_option_list[0])

        psi0_label = tk.Label(self.top_right, text='Initial Wavefunction')
        self.psi0_optionsmenu = tk.OptionMenu(self.top_right, self.psi0_option_sv, *self.psi0_option_list)
        self.psi0_option_sv.trace_add("write", self.changeFramePsi0)

        psi0_label.grid(row=2, column=0)
        self.psi0_optionsmenu.grid(row=2, column=1, columnspan=5)
        ## set up frames for each option in the dropdown menu
        ## thomas fermi
        self.psi0_thomas_fermi_frame = tk.Frame(self.top_right, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.psi0_thomas_fermi_frame.grid(row=3, column=0, columnspan=5, padx=self.padx, pady=self.pady)

        psi0_thomas_fermi_gamma_label = tk.Label(self.psi0_thomas_fermi_frame, text='Gamma y')
        self.psi0_thomas_fermi_gamma_sv = tk.DoubleVar(value=self.paramObj.psi0_parameters["gamma_y"])
        self.psi0_thomas_fermi_gamma_sv.trace_add("write", self.onChange)
        self.psi0_thomas_fermi_gamma_scale = tk.Scale(self.psi0_thomas_fermi_frame, from_=0.1, to_=3, resolution=0.1, tickinterval=0.9, orient=tk.HORIZONTAL, variable=self.psi0_thomas_fermi_gamma_sv, length=150)

        psi0_thomas_fermi_gamma_label.grid(row=0, column=0)
        self.psi0_thomas_fermi_gamma_scale.grid(row=0, column=1)

        ## gauss
        self.psi0_gauss_frame = tk.Frame(self.top_right, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.psi0_gauss_frame.grid(row=3, column=0, columnspan=5, padx=self.padx, pady=self.pady)

        psi0_gauss_sigma_label = tk.Label(self.psi0_gauss_frame, text='Sigma')
        self.psi0_gauss_sigma_sv = tk.DoubleVar(value=self.paramObj.psi0_parameters["sigma"])
        self.psi0_gauss_sigma_sv.trace_add("write", self.onChange)
        self.psi0_gauss_sigma_scale = tk.Scale(self.psi0_gauss_frame, from_=0.1, to_=3, resolution=0.1, tickinterval=0.9, orient=tk.HORIZONTAL, variable=self.psi0_gauss_sigma_sv, length=200)

        psi0_gauss_x0_label = tk.Label(self.psi0_gauss_frame, text='x0')
        self.psi0_gauss_x0_sv = tk.DoubleVar(value=self.paramObj.psi0_parameters["x0"])
        self.psi0_gauss_x0_sv.trace_add("write", self.onChange)
        self.psi0_gauss_x0_scale = tk.Scale(self.psi0_gauss_frame, from_=-10, to_=10, resolution=0.5, tickinterval=5, orient=tk.HORIZONTAL, variable=self.psi0_gauss_x0_sv, length=150)

        psi0_gauss_y0_label = tk.Label(self.psi0_gauss_frame, text='y0')
        self.psi0_gauss_y0_sv = tk.DoubleVar(value=self.paramObj.psi0_parameters["y0"])
        self.psi0_gauss_y0_sv.trace_add("write", self.onChange)
        self.psi0_gauss_y0_scale = tk.Scale(self.psi0_gauss_frame, from_=-10, to_=10, resolution=0.5, tickinterval=5, orient=tk.HORIZONTAL, variable=self.psi0_gauss_y0_sv, length=150)

        psi0_gauss_sigma_label.grid(row=0, column=0)
        self.psi0_gauss_sigma_scale.grid(row=0, column=1, columnspan=3)

        psi0_gauss_x0_label.grid(row=1, column=0)
        self.psi0_gauss_x0_scale.grid(row=1, column=1)

        psi0_gauss_y0_label.grid(row=1, column=2)
        self.psi0_gauss_y0_scale.grid(row=1, column=3)

        self.psi0_thomas_fermi_frame.grid_remove()
        self.psi0_gauss_frame.grid_remove()

        self.psi0_thomas_fermi_frame.grid()

        # dropdown menu for the Potential V
        self.V_option_list = ('Harmonic', 'Harmonic + Quartic', 'Harmonic + Optic')
        self.V_option_sv = tk.StringVar(value=self.V_option_list[0])
        self.V_option_sv.trace_add("write", self.changeFrameV)

        V_label = tk.Label(self.top_right, text='Potential V')
        self.V_optionsmenu = tk.OptionMenu(self.top_right, self.V_option_sv, *self.V_option_list)

        V_label.grid(row=4, column=0)
        self.V_optionsmenu.grid(row=4, column=1, columnspan=4)

        ## refresh button
        self.V_refresh_button = tk.Button(self.top_right, command=self.updatePlot, text='Refresh')
        self.V_refresh_button.grid(row=6, column=5)

        ## set up frames for each option in the dropdown menu
        ## V harmonic
        self.V_harmonic_frame = tk.Frame(self.top_right, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.V_harmonic_frame.grid(row=5, column=0, columnspan=5, padx=self.padx, pady=self.pady)

        V_harmonic_gamma_label = tk.Label(self.V_harmonic_frame, text='Gamma y')
        self.V_harmonic_gamma_sv = tk.DoubleVar(value=self.paramObj.potential_parameters["gamma_y"])
        self.V_harmonic_gamma_sv.trace_add("write", self.onChange)
        self.V_harmonic_gamma_scale = tk.Scale(self.V_harmonic_frame, from_=0.1, to_=3, resolution=0.1, tickinterval=0.9, orient=tk.HORIZONTAL, variable=self.V_harmonic_gamma_sv, length=150)

        V_harmonic_gamma_label.grid(row=0, column=0)
        self.V_harmonic_gamma_scale.grid(row=0, column=1)

        ## harmonic quartic
        self.V_harmonic_quartic_frame = tk.Frame(self.top_right, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.V_harmonic_quartic_frame.grid(row=5, column=0, columnspan=5, padx=self.padx, pady=self.pady)

        V_harmonic_quartic_alpha_label = tk.Label(self.V_harmonic_quartic_frame, text='Alpha')
        self.V_harmonic_quartic_alpha_sv = tk.DoubleVar(value=self.paramObj.potential_parameters["alpha"])
        self.V_harmonic_quartic_alpha_sv.trace_add("write", self.onChange)
        self.V_harmonic_quartic_alpha_scale = tk.Scale(self.V_harmonic_quartic_frame, from_=0, to_=3, resolution=0.1, tickinterval=1, orient=tk.HORIZONTAL, variable=self.V_harmonic_quartic_alpha_sv, length=150)

        V_harmonic_quartic_kappa_label = tk.Label(self.V_harmonic_quartic_frame, text='Kappa')
        self.V_harmonic_quartic_kappa_sv = tk.DoubleVar(value=self.paramObj.potential_parameters["kappa_quartic"])
        self.V_harmonic_quartic_kappa_sv.trace_add("write", self.onChange)
        self.V_harmonic_quartic_kappa_scale = tk.Scale(self.V_harmonic_quartic_frame, from_=0, to_=3, resolution=0.1, tickinterval=1, orient=tk.HORIZONTAL, variable=self.V_harmonic_quartic_kappa_sv, length=150)

        V_harmonic_quartic_alpha_label.grid(row=0, column=0)
        self.V_harmonic_quartic_alpha_scale.grid(row=0, column=1)
        
        V_harmonic_quartic_kappa_label.grid(row=0, column=2)
        self.V_harmonic_quartic_kappa_scale.grid(row=0, column=3)

        ## harmonic optic
        self.V_harmonic_optic_frame = tk.Frame(self.top_right, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.V_harmonic_optic_frame.grid(row=5, column=0, columnspan=5, padx=self.padx, pady=self.pady)

        V_harmonic_optic_v0_label = tk.Label(self.V_harmonic_optic_frame, text='V0')
        self.V_harmonic_optic_v0_sv = tk.DoubleVar(value=self.paramObj.potential_parameters["V0"])
        self.V_harmonic_optic_v0_sv.trace_add("write", self.onChange)
        self.V_harmonic_optic_v0_scale = tk.Scale(self.V_harmonic_optic_frame, from_=0, to_=50, resolution=1, tickinterval=25, orient=tk.HORIZONTAL, variable=self.V_harmonic_optic_v0_sv, length=150)

        V_harmonic_optic_kappa_label = tk.Label(self.V_harmonic_optic_frame, text='Kappa')
        self.V_harmonic_optic_kappa_sv = tk.DoubleVar(value=self.paramObj.potential_parameters["kappa_optic"])
        self.V_harmonic_optic_kappa_sv.trace_add("write", self.onChange)
        self.V_harmonic_optic_kappa_scale = tk.Scale(self.V_harmonic_optic_frame, from_=0, to_=5, resolution=0.1, tickinterval=1, orient=tk.HORIZONTAL, variable=self.V_harmonic_optic_kappa_sv, length=150)

        V_harmonic_optic_v0_label.grid(row=0, column=0)
        self.V_harmonic_optic_v0_scale.grid(row=0, column=1)
        
        V_harmonic_optic_kappa_label.grid(row=0, column=2)
        self.V_harmonic_optic_kappa_scale.grid(row=0, column=3)

        self.V_harmonic_frame.grid_remove()
        self.V_harmonic_optic_frame.grid_remove()
        self.V_harmonic_quartic_frame.grid_remove()

        self.V_harmonic_frame.grid()

    def init_bottom_left(self):
        # initializes the bottom left frame which contains the numerical parameters
        # set up slider for dt
        dt_label = tk.Label(self.bottom_left, text='Delta t')
        self.dt_sv = tk.DoubleVar(value=self.paramObj.dt)
        self.dt_sv.trace_add("write", self.onChange)
        self.dt_scale = tk.Scale(self.bottom_left, from_=0, to_=0.01, resolution=0.001, tickinterval=0.005, orient=tk.HORIZONTAL, variable=self.dt_sv, length=150)
        self.dt_constraint_label = tk.Label(self.bottom_left, text='has to be smaller than {:1.2e}'.format(self.dt_constraint))

        # slider for epsilon limit
        epsilon_label = tk.Label(self.bottom_left, text='Epsilon Limit Exponent')
        epsilon_label2 = tk.Label(self.bottom_left, text='This is an exit condition on the simulation.\nIf the difference of two adjacent frames is smaller than\n10 to the power of this exponent, the simulation terminates.')
        self.epsilon_sv = tk.IntVar(value=-10)
        self.epsilon_sv.trace_add("write", self.onChange)
        self.epsilon_scale = tk.Scale(self.bottom_left, from_=-3, to_=-14, resolution=1, tickinterval=3, orient=tk.HORIZONTAL, variable=self.epsilon_sv, length=150)

        # slider for max Iterations
        maxIter_label = tk.Label(self.bottom_left, text='max. number of Iterations')
        self.maxIter_sv = tk.IntVar(value=self.paramObj.maxIterations)
        self.maxIter_sv.trace_add("write", self.onChange)
        self.maxIter_scale = tk.Scale(self.bottom_left, from_=10_000, to_=600_000, resolution=10_000, tickinterval=200_000, orient=tk.HORIZONTAL, variable=self.maxIter_sv, length=250)

        # field for filename, including a button
        filename_label = tk.Label(self.bottom_left, text='Filename')
        self.filename_sv = tk.StringVar(value=self.paramObj.filename)
        self.filename_entry = tk.Entry(self.bottom_left, width=30, justify=tk.RIGHT, textvariable=self.filename_sv)
        self.filename_entry['state'] = tk.DISABLED
        self.filename_button = tk.Button(self.bottom_left, text='Choose Filename', command=self.chooseFile)

        # slider for epsilon threshold
        epsilon_threshold_label = tk.Label(self.bottom_left, text='Epsilon Threshold')
        epsilon_threshold_label2 = tk.Label(self.bottom_left, text='Relative amount of Frames saved during simulation.\nA lower number results in more saved frames and a larger filesize.')
        self.epsilon_threshold_sv = tk.DoubleVar(value=1)
        self.epsilon_threshold_sv.trace_add("write", self.onChange)
        self.epsilon_threshold_scale = tk.Scale(self.bottom_left, from_=0, to_=2, resolution=0.1, tickinterval=1, orient=tk.HORIZONTAL, variable=self.epsilon_threshold_sv, length=150)

        # place them all on the window
        dt_label.grid(row=0, column=0)
        self.dt_scale.grid(row=0, column=1)
        self.dt_constraint_label.grid(row=0, column=2)

        epsilon_label.grid(row=1, column=0)
        self.epsilon_scale.grid(row=1, column=1)
        epsilon_label2.grid(row=1, column=2, columnspan=2)

        maxIter_label.grid(row=2, column=0)
        self.maxIter_scale.grid(row=2, column=1, columnspan=2)

        filename_label.grid(row=3, column=0)
        self.filename_entry.grid(row=3, column=1, columnspan=2)
        self.filename_button.grid(row=3, column=3)

        epsilon_threshold_label.grid(row=4, column=0)
        self.epsilon_threshold_scale.grid(row=4, column=1)
        epsilon_threshold_label2.grid(row=4, column=2, columnspan=2)

        ## start simulation button
        font_large = font.Font(family='Helvetica', size=25, weight='normal')
        self.start_button = tk.Button(self.bottom_left, command=self.startCalculation, text='Start Simulation', width=15, height=1, font=font_large, bg='red')
        self.start_button.grid(row=5, column=0, columnspan=5)

    def init_bottom_right(self):
        # initializes the bottom right frame which contains the preview plot of the trapping potential
        # set up matplotlib figure
        fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        plt.tight_layout()
        self.im = self.ax.imshow(self.paramObj.V, cmap='jet')
        self.cb = fig.colorbar(self.im)

        # set up axes labels
        self.ax.set_title("Potential V")
        self.ax.set_xlabel("$x$")
        self.ax.set_xticks(np.linspace(0, 255, 5))
        self.ax.set_xticklabels(np.linspace(-16, 16, 5))
        self.ax.set_ylabel("$y$")
        self.ax.set_yticks(np.linspace(0, 255, 5))
        self.ax.set_yticklabels(np.linspace(-16, 16, 5))

        # set up integration with tkinter
        self.V_canvas = FigureCanvasTkAgg(fig, master=self.bottom_right)  # A tk.DrawingArea.
        self.V_canvas.draw()
        self.V_canvas.get_tk_widget().grid(row=0, column=0)

    def changeFramePsi0(self, a, b, c):
        # this function is calles when the user selects an option in the initial wavefunction dropdown menu
        choice = self.psi0_option_sv.get()
        # switch to the correct frame and display it
        if choice == self.psi0_option_list[0]:
            self.psi0_gauss_frame.grid_remove()
            self.psi0_thomas_fermi_frame.grid()
            self.paramObj.psi0_choice = Psi0Choice.THOMAS_FERMI
        elif choice == self.psi0_option_list[1]:
            self.psi0_thomas_fermi_frame.grid_remove()
            self.psi0_gauss_frame.grid()
            self.paramObj.psi0_choice = Psi0Choice.GAUSS
        else:
            messagebox.showerror("ERROR", "Choice for Psi0 not recognized, may not be implemented...")
            self.paramObj.choice_psi0 = Psi0Choice.NOT_IMPLEMENTED
        # apply changes to update potential preview
        if self.applyChanges():
            self.paramObj.initV()
            self.updatePlot()
            self.update_dt_constraint()

    def changeFrameV(self, a, b, c):
        # this function is calles when the user selects an option in the trapping potential dropdown menu
        choice = self.V_option_sv.get()
        # switch to the correct frame and display it
        if choice == self.V_option_list[0]:
            self.V_harmonic_optic_frame.grid_remove()
            self.V_harmonic_quartic_frame.grid_remove()
            self.V_harmonic_frame.grid()
            self.paramObj.potential_choice = PotentialChoice.HARMONIC
        elif choice == self.V_option_list[1]:
            self.V_harmonic_frame.grid_remove()
            self.V_harmonic_optic_frame.grid_remove()
            self.V_harmonic_quartic_frame.grid()
            self.paramObj.potential_choice = PotentialChoice.HARMONIC_QUARTIC
        elif choice == self.V_option_list[2]:
            self.V_harmonic_quartic_frame.grid_remove()
            self.V_harmonic_frame.grid_remove()
            self.V_harmonic_optic_frame.grid()
            self.paramObj.potential_choice = PotentialChoice.HARMONIC_OPTIC
        else:
            messagebox.showerror("ERROR", "Choice for potential V not recognized, may not be implemented...")
            self.paramObj.potential_choice = PotentialChoice.NOT_IMPLEMENTED

        # apply changes to update potential preview
        if self.applyChanges():
            self.paramObj.initV()
            self.updatePlot()
            self.update_dt_constraint()

    def applyChanges(self):
        # this function collects all user set values and bundles them in a parameter object

        # top left
        self.paramObj.x_low = self.x_low_sv.get()
        self.paramObj.x_high = self.x_high_sv.get()
        self.paramObj.y_low = self.y_low_sv.get()
        self.paramObj.y_high = self.y_high_sv.get()

        self.paramObj.resolutionX = self.res_x_sv.get()
        self.paramObj.resolutionY = self.res_y_sv.get()


        # bottom left
        self.paramObj.dt = self.dt_sv.get()
        self.paramObj.epsilon_limit = 10**(self.epsilon_sv.get())
        self.paramObj.maxIterations = self.maxIter_sv.get()
        self.paramObj.filename = self.filename_sv.get()
        self.paramObj.epsilon_threshold = self.epsilon_threshold_sv.get() * 1/(200*self.paramObj.dt)

        # top right
        self.paramObj.omega = self.omega_sv.get()
        self.paramObj.beta2 = self.beta_sv.get()

        # # Psi 0
        if self.paramObj.psi0_choice == Psi0Choice.THOMAS_FERMI:
            self.paramObj.psi0_parameters['gamma_y'] = self.psi0_thomas_fermi_gamma_sv.get()
        elif self.paramObj.psi0_choice == Psi0Choice.GAUSS:
            self.paramObj.psi0_parameters['sigma'] = self.psi0_gauss_sigma_sv.get()
            self.paramObj.psi0_parameters['x0'] = self.psi0_gauss_y0_sv.get()
            self.paramObj.psi0_parameters['y0'] = self.psi0_gauss_y0_sv.get()

        # # potential V
        if self.paramObj.potential_choice == PotentialChoice.HARMONIC:
            self.paramObj.potential_parameters["gamma_y"] = self.V_harmonic_gamma_sv.get()
        elif self.paramObj.potential_choice == PotentialChoice.HARMONIC_QUARTIC:
                self.paramObj.potential_parameters["alpha"] = self.V_harmonic_quartic_alpha_sv.get()
                self.paramObj.potential_parameters["kappa_quartic"] = self.V_harmonic_quartic_kappa_sv.get()
        elif self.paramObj.potential_choice == PotentialChoice.HARMONIC_OPTIC:
                self.paramObj.potential_parameters["V0"] = self.V_harmonic_optic_v0_sv.get()
                self.paramObj.potential_parameters["kappa_optic"] = self.V_harmonic_optic_kappa_sv.get()
        else:
            messagebox.showerror("ERROR", "Potential not recognized")
            return False

        return True

    def chooseFile(self):
        # this function is called when the user presses the select filename button
        # it produces a file menu
        filetypes = [('hdf5 database files', '*.hdf5'), ('All files', '*')]
        filename = asksaveasfilename(title='Save as..', defaultextension='.hdf5', filetypes=filetypes)
        if filename:
            self.filename_sv.set(filename)

    def updatePlot(self):
        # this function updates the plot according to the updated values in the parameter object
        self.ax.set_xticklabels(np.linspace(self.paramObj.x_low, self.paramObj.x_high, 5))
        self.ax.set_yticklabels(np.linspace(self.paramObj.y_low, self.paramObj.y_high, 5))

        self.im.set_data(self.paramObj.V)
        self.cb.set_clim(np.min(self.paramObj.V), np.max(self.paramObj.V))
        self.cb.set_ticks(np.linspace(np.min(self.paramObj.V), np.max(self.paramObj.V), 6))
        self.cb.draw_all()
        self.V_canvas.draw()

    def onChange(self, a, b, c):
        # this function is called whenever any slider is moved.
        if self.applyChanges():
            self.paramObj.initV()
            self.updatePlot()
        self.update_dt_constraint()
        
    def calc_dt_constraint(self):
        # this function calculates the time step constraint for the numerical method used
        # see https://doi.org/10.1016/j.jcp.2006.04.019 p.8
        w = WaveFunction2D(self.paramObj)
        w.initPsi_0()
        b_ = self.paramObj.V + self.paramObj.beta2*np.abs(w.psi_array)**2
        bmin = np.min(b_)
        bmax = np.max(b_)
        self.dt_constraint = 2/(bmax+bmin)

    def update_dt_constraint(self):
        # updates the max. value for the slider that controls dt
        self.calc_dt_constraint()
        if self.dt_sv.get() > self.dt_constraint:
            self.dt_sv.set(self.dt_constraint/2)
        self.dt_scale.config(from_=self.dt_constraint/20)
        self.dt_scale.config(to_=self.dt_constraint*(19/20))
        self.dt_scale.config(resolution = self.dt_constraint/10)
        self.dt_scale.config(tickinterval = self.dt_constraint/2)
        self.dt_constraint_label['text'] = 'has to be smaller than {:1.2e}'.format(self.dt_constraint)

    def focusOut(self, a):
        # i dont think this function is used anymore
        # does the same thing as onChange()
        if self.applyChanges():
            self.paramObj.initV()
            self.updatePlot()
        self.update_dt_constraint()

    def applyPreset(self, po):
        # this function applies the settings contained in parameter object po to the GUI

        # top left
        self.x_low_sv.set(po.x_low)
        self.x_high_sv.set(po.x_high)
        self.y_low_sv.set(po.y_low)
        self.y_high_sv.set(po.y_high)
        self.res_x_sv.set(po.resolutionX)
        self.res_y_sv.set(po.resolutionY)

        # top right
        self.omega_sv.set(po.omega)
        self.beta_sv.set(po.beta2)
        if po.psi0_choice == Psi0Choice.THOMAS_FERMI:
            self.psi0_option_sv.set(self.psi0_option_list[0])
            self.psi0_thomas_fermi_gamma_sv.set(po.psi0_parameters['gamma_y'])
        elif po.psi0_choice == Psi0Choice.GAUSS:
            self.psi0_option_sv.set(self.psi0_option_list[1])
            self.psi0_gauss_sigma_sv.set(po.psi0_parameters['sigma'])
            self.psi0_gauss_x0_sv.set(po.psi0_parameters['x0'])
            self.psi0_gauss_y0_sv.set(po.psi0_parameters['y0'])

        if po.potential_choice == PotentialChoice.HARMONIC:
            self.V_option_sv.set(self.V_option_list[0])
            self.V_harmonic_gamma_sv.set(po.potential_parameters["gamma_y"])
        elif po.potential_choice == PotentialChoice.HARMONIC_QUARTIC:
            self.V_option_sv.set(self.V_option_list[1])
            self.V_harmonic_quartic_alpha_sv.set(po.potential_parameters["alpha"])
            self.V_harmonic_quartic_kappa_sv.set(po.potential_parameters["kappa_quartic"])
        elif po.potential_choice == PotentialChoice.HARMONIC_OPTIC:
            self.V_option_sv.set(self.V_option_list[2])
            self.V_harmonic_optic_v0_sv.set(po.potential_parameters["V0"])
            self.V_harmonic_optic_kappa_sv.set(po.potential_parameters["kappa_optic"])

        # bottom left
        self.dt_sv.set(po.dt)
        self.maxIter_sv.set(po.maxIterations)
        self.epsilon_sv.set(int(np.log10(po.epsilon_limit)))

    def loadPresetNoRotation(self):
        # defines a preset where no rotation is persent
        po = ParameterObject(resolutionX = 256, resolutionY = 256,
                            x_low = -16, x_high = 16, y_low = -16, y_high = 16,
                            beta2 = 1000, omega = 0,
                            epsilon_limit=1e-8, epsilon_threshold=1, dt=0.005, maxIterations=30_000,
                            filename='default.hdf5',
                            potential_choice=PotentialChoice.HARMONIC,
                            potential_parameters={'gamma_y':1, 'alpha':1.2, 'kappa_quartic':0.3, 'kappa_optic':0.7, 'V0':5},
                            psi0_choice=Psi0Choice.GAUSS, psi0_parameters={'gamma_y':1, 'sigma':1, 'x0':5, 'y0':5})
        self.applyPreset(po)

    def loadPresetBetaSmall(self):
        # defines a preset where rotation is present and the self interaction of the BEC is weak
        po = ParameterObject(resolutionX = 256, resolutionY = 256,
                            x_low = -16, x_high = 16, y_low = -16, y_high = 16,
                            beta2 = 100, omega = 0.85,
                            epsilon_limit=1e-10, epsilon_threshold=1, dt=0.005, maxIterations=40_000,
                            filename='default.hdf5',
                            potential_choice=PotentialChoice.HARMONIC,
                            potential_parameters={'gamma_y':1, 'alpha':1.2, 'kappa_quartic':0.3, 'kappa_optic':0.7, 'V0':5},
                            psi0_choice=Psi0Choice.THOMAS_FERMI, psi0_parameters={'gamma_y':1, 'sigma':1, 'x0':0, 'y0':0})
        self.applyPreset(po)

    def loadPresetBetaLarge(self):
        # defines a preset where rotation is present and the self interaction of the BEC is strong
        po = ParameterObject(resolutionX = 256, resolutionY = 256,
                            x_low = -16, x_high = 16, y_low = -16, y_high = 16,
                            beta2 = 1000, omega = 0.85,
                            epsilon_limit=1e-10, epsilon_threshold=1, dt=0.005, maxIterations=80_000,
                            filename='default.hdf5',
                            potential_choice=PotentialChoice.HARMONIC,
                            potential_parameters={'gamma_y':1, 'alpha':1.2, 'kappa_quartic':0.3, 'kappa_optic':0.7, 'V0':5},
                            psi0_choice=Psi0Choice.THOMAS_FERMI, psi0_parameters={'gamma_y':1, 'sigma':1, 'x0':0, 'y0':0})
        self.applyPreset(po)

    def loadPresetAnisotropic(self):
        # defines a preset where the trapping potential is an anisotropic harmonic oscillator
        po = ParameterObject(resolutionX = 256, resolutionY = 256,
                            x_low = -16, x_high = 16, y_low = -16, y_high = 16,
                            beta2 = 1000, omega = 0.7,
                            epsilon_limit=1e-8, epsilon_threshold=1, dt=0.005, maxIterations=70_000,
                            filename='default.hdf5',
                            potential_choice=PotentialChoice.HARMONIC,
                            potential_parameters={'gamma_y':0.5, 'alpha':1.2, 'kappa_quartic':0.3, 'kappa_optic':0.7, 'V0':5},
                            psi0_choice=Psi0Choice.THOMAS_FERMI, psi0_parameters={'gamma_y':0.5, 'sigma':1, 'x0':0, 'y0':0})
        self.applyPreset(po)

    def loadPresetHarmonicQuartic(self):
        # defines a preset where a harmonic + quartic trapping potential is present and the rotation frequency is over critical (larger than 1)
        po = ParameterObject(resolutionX = 512, resolutionY = 512,
                            x_low = -16, x_high = 16, y_low = -16, y_high = 16,
                            beta2 = 1000, omega = 2,
                            epsilon_limit=1e-8, epsilon_threshold=1, dt=0.005, maxIterations=500_000,
                            filename='default.hdf5',
                            potential_choice=PotentialChoice.HARMONIC_QUARTIC,
                            potential_parameters={'gamma_y':1, 'alpha':1.2, 'kappa_quartic':0.3, 'kappa_optic':0.7, 'V0':5},
                            psi0_choice=Psi0Choice.THOMAS_FERMI, psi0_parameters={'gamma_y':1, 'sigma':1, 'x0':0, 'y0':0})
        self.applyPreset(po)

    def loadPresetHarmonicOptic(self):
        # defines a preset where an harmonic plus optic potential is present
        po = ParameterObject(resolutionX = 256, resolutionY = 256,
                            x_low = -16, x_high = 16, y_low = -16, y_high = 16,
                            beta2 = 200, omega = 0.9,
                            epsilon_limit=1e-8, epsilon_threshold=1, dt=0.005, maxIterations=100_000,
                            filename='default.hdf5',
                            potential_choice=PotentialChoice.HARMONIC_OPTIC,
                            potential_parameters={'gamma_y':1, 'alpha':1.2, 'kappa_quartic':0.3, 'kappa_optic':0.7, 'V0':10},
                            psi0_choice=Psi0Choice.THOMAS_FERMI, psi0_parameters={'gamma_y':1, 'sigma':1, 'x0':0, 'y0':0})
        self.applyPreset(po)

    def startCalculation(self):
        # this function is called when the user presses the big start simulation button
        # does what you think it does :)
        print("[INFO] Start of Simulation.")
        if self.applyChanges():
            self.update_dt_constraint()
            if self.dt_constraint < self.paramObj.dt:
                messagebox.showerror("Parameter-Fehler dt", "Der Zeitschritt dt muss kleiner sein als der angezeigte Maximalwert!")
            else:
                self.paramObj.initV()
                self.parent.destroy()
                self.pressedStart = True



if __name__ == "__main__":
    # this code should not ever be excecuted and is only here for debug purposes
    root = tk.Tk()
    ParameterApp(root)
    root.mainloop()