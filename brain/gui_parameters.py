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
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # set up standard parameters
        self.paramObj = ParameterObject()
        self.paramObj.initV()

        self.dt_constraint = None
        self.calc_dt_constraint()

        self.pressedStart = False

        self.padx = 5
        self.pady = 5

        self.init_frames()
        self.init_top_left()
        self.init_top_right()
        self.init_bottom_left()
        self.init_bottom_right()

    def init_frames(self):
        self.topLevelFrame = tk.Frame(self.parent)
        self.topLevelFrame.grid()

        self.top_left = tk.LabelFrame(self.topLevelFrame, text='Raster-Parameter', labelanchor='nw', width=400, height=400, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.top_right = tk.LabelFrame(self.topLevelFrame, text='Physikalische Parameter', labelanchor='nw', width=400, height=400, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.bottom_left = tk.LabelFrame(self.topLevelFrame, text='Numerische Parameter', labelanchor='nw', width=400, height=400, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.bottom_right = tk.LabelFrame(self.topLevelFrame, text='Potential V', labelanchor='nw', width=400, height=400, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        

        # layout all of the main containers
        # self.parent.grid_rowconfigure(1, weight=1)
        # self.parent.grid_columnconfigure(0, weight=1)

        self.top_left.grid(row=0, column=0, padx=self.padx, pady=self.pady, sticky='')
        self.top_right.grid(row=0, column=1, padx=self.padx, pady=self.pady, sticky='')
        self.bottom_left.grid(row=1, column=0, padx=self.padx, pady=self.pady, sticky='')
        self.bottom_right.grid(row=1, column=1, padx=self.padx, pady=self.pady, sticky='')

    def init_top_left(self):
        # grid spacing parameters
        x_low_label = tk.Label(self.top_left, text='x low')
        self.x_low_sv = tk.StringVar(value=self.paramObj.x_low)
        self.x_low_sv.trace_add("write", self.onChange)
        self.x_low_entry = tk.Entry(self.top_left, width=10, justify=tk.RIGHT, textvariable=self.x_low_sv)
        self.x_low_entry.bind("<FocusOut>", self.focusOut)
        self.x_low_entry.bind("<Return>", self.focusOut)

        x_high_label = tk.Label(self.top_left, text='x high')
        self.x_high_sv = tk.StringVar(value=self.paramObj.x_high)
        self.x_high_sv.trace_add("write", self.onChange)
        self.x_high_entry = tk.Entry(self.top_left, width=10, justify=tk.RIGHT, textvariable=self.x_high_sv)
        self.x_high_entry.bind("<FocusOut>", self.focusOut)
        self.x_high_entry.bind("<Return>", self.focusOut)

        y_low_label = tk.Label(self.top_left, text='y low')
        self.y_low_sv = tk.StringVar(value=self.paramObj.y_low)
        self.y_low_sv.trace_add("write", self.onChange)
        self.y_low_entry = tk.Entry(self.top_left, width=10, justify=tk.RIGHT, textvariable=self.y_low_sv)
        self.y_low_entry.bind("<FocusOut>", self.focusOut)
        self.y_low_entry.bind("<Return>", self.focusOut)

        y_high_label = tk.Label(self.top_left, text='y high')
        self.y_high_sv = tk.StringVar(value=self.paramObj.y_high)
        self.y_high_sv.trace_add("write", self.onChange)
        self.y_high_entry = tk.Entry(self.top_left, width=10, justify=tk.RIGHT,textvariable=self.y_high_sv)
        self.y_high_entry.bind("<FocusOut>", self.focusOut)
        self.y_high_entry.bind("<Return>", self.focusOut)

        # self.top_left.grid_columnconfigure(2, minsize=80)

        x_low_label.grid(row=0, column=0)
        self.x_low_entry.grid(row=0, column=1)

        x_high_label.grid(row=0, column=3)
        self.x_high_entry.grid(row=0, column=4)

        y_low_label.grid(row=1, column=0)
        self.y_low_entry.grid(row=1, column=1)

        y_high_label.grid(row=1, column=3)
        self.y_high_entry.grid(row=1, column=4)

        # resolution
        color_red = '#ff8080'
        self.resolution_frame = tk.Frame(self.top_left, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3, bg=color_red)
        self.resolution_frame.grid(row=2, column=0, columnspan=5, padx=self.padx, pady=self.pady)

        res_x_label = tk.Label(self.resolution_frame, text='Auflösung x-Richtung', bg=color_red)
        self.res_x_sv = tk.StringVar(value=self.paramObj.resolutionX)
        self.res_x_sv.trace_add("write", self.onChange)
        self.res_x_entry = tk.Entry(self.resolution_frame, width=10, justify=tk.RIGHT, textvariable=self.res_x_sv)
        self.res_x_entry.bind("<FocusOut>", self.focusOut)
        self.res_x_entry.bind("<Return>", self.focusOut)

        res_y_label = tk.Label(self.resolution_frame, text='Auflösung y-Richtung', bg=color_red)
        self.res_y_sv = tk.StringVar(value=self.paramObj.resolutionY)
        self.res_y_sv.trace_add("write", self.onChange)
        self.res_y_entry = tk.Entry(self.resolution_frame, width=10, justify=tk.RIGHT, textvariable=self.res_y_sv)
        self.res_y_entry.bind("<FocusOut>", self.focusOut)
        self.res_y_entry.bind("<Return>", self.focusOut)

        res_x_label.grid(row=0, column=0)
        self.res_x_entry.grid(row=0, column=1)

        res_y_label.grid(row=1, column=0)
        self.res_y_entry.grid(row=1, column=1)

    def init_top_right(self):
        # omega and beta
        omega_label = tk.Label(self.top_right, text='Omega')
        self.omega_sv = tk.StringVar(value=self.paramObj.omega)
        self.omega_sv.trace_add("write", self.onChange)
        self.omega_entry = tk.Entry(self.top_right, width=10, justify=tk.RIGHT, textvariable=self.omega_sv)
        self.omega_entry.bind("<FocusOut>", self.focusOut)
        self.omega_entry.bind("<Return>", self.focusOut)

        beta_label = tk.Label(self.top_right, text='Beta')
        self.beta_sv = tk.StringVar(value=self.paramObj.beta2)
        self.beta_sv.trace_add("write", self.onChange)
        self.beta_entry = tk.Entry(self.top_right, width=10, justify=tk.RIGHT, textvariable=self.beta_sv)
        self.beta_entry.bind("<FocusOut>", self.focusOut)
        self.beta_entry.bind("<Return>", self.focusOut)

        omega_label.grid(row=0, column=0)
        self.omega_entry.grid(row=0, column=1)

        beta_label.grid(row=0, column=3)
        self.beta_entry.grid(row=0, column=4)

        # Psi 0
        self.psi0_option_list = ('Thomas-Fermi-Approximation', 'Gauss')
        self.psi0_option_sv = tk.StringVar(value=self.psi0_option_list[0])

        psi0_label = tk.Label(self.top_right, text='Start-Wellenfunktion')
        self.psi0_optionsmenu = tk.OptionMenu(self.top_right, self.psi0_option_sv, *self.psi0_option_list)
        self.psi0_option_sv.trace_add("write", self.changeFramePsi0)

        psi0_label.grid(row=1, column=0)
        self.psi0_optionsmenu.grid(row=1, column=1, columnspan=5)

        # # thomas fermi
        self.psi0_thomas_fermi_frame = tk.Frame(self.top_right, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.psi0_thomas_fermi_frame.grid(row=2, column=0, columnspan=5, padx=self.padx, pady=self.pady)

        psi0_thomas_fermi_gamma_label = tk.Label(self.psi0_thomas_fermi_frame, text='Gamma y')
        self.psi0_thomas_fermi_gamma_sv = tk.StringVar(value=self.paramObj.psi0_parameters["gamma_y"])
        self.psi0_thomas_fermi_gamma_entry = tk.Entry(self.psi0_thomas_fermi_frame, width=10, justify=tk.RIGHT, textvariable=self.psi0_thomas_fermi_gamma_sv)
        self.psi0_thomas_fermi_gamma_entry.bind("<FocusOut>", self.focusOut)
        self.psi0_thomas_fermi_gamma_entry.bind("<Return>", self.focusOut)


        psi0_thomas_fermi_gamma_label.grid(row=0, column=0)
        self.psi0_thomas_fermi_gamma_entry.grid(row=0, column=1)

        # # gauss
        self.psi0_gauss_frame = tk.Frame(self.top_right, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.psi0_gauss_frame.grid(row=2, column=0, columnspan=5, padx=self.padx, pady=self.pady)

        psi0_gauss_sigma_label = tk.Label(self.psi0_gauss_frame, text='Sigma')
        self.psi0_gauss_sigma_sv = tk.StringVar(value=self.paramObj.psi0_parameters["sigma"])
        self.psi0_gauss_sigma_entry = tk.Entry(self.psi0_gauss_frame, width=10, justify=tk.RIGHT, textvariable=self.psi0_gauss_sigma_sv)
        self.psi0_gauss_sigma_entry.bind("<FocusOut>", self.focusOut)
        self.psi0_gauss_sigma_entry.bind("<Return>", self.focusOut)

        psi0_gauss_x0_label = tk.Label(self.psi0_gauss_frame, text='x0')
        self.psi0_gauss_x0_sv = tk.StringVar(value=self.paramObj.psi0_parameters["x0"])
        self.psi0_gauss_x0_entry = tk.Entry(self.psi0_gauss_frame, width=10, justify=tk.RIGHT, textvariable=self.psi0_gauss_x0_sv)
        self.psi0_gauss_x0_entry.bind("<FocusOut>", self.focusOut)
        self.psi0_gauss_x0_entry.bind("<Return>", self.focusOut)

        psi0_gauss_y0_label = tk.Label(self.psi0_gauss_frame, text='y0')
        self.psi0_gauss_y0_sv = tk.StringVar(value=self.paramObj.psi0_parameters["y0"])
        self.psi0_gauss_y0_entry = tk.Entry(self.psi0_gauss_frame, width=10, justify=tk.RIGHT, textvariable=self.psi0_gauss_y0_sv)
        self.psi0_gauss_y0_entry.bind("<FocusOut>", self.focusOut)
        self.psi0_gauss_y0_entry.bind("<Return>", self.focusOut)

        psi0_gauss_sigma_label.grid(row=0, column=0)
        self.psi0_gauss_sigma_entry.grid(row=0, column=1)

        psi0_gauss_x0_label.grid(row=1, column=0)
        self.psi0_gauss_x0_entry.grid(row=1, column=1)

        psi0_gauss_y0_label.grid(row=2, column=0)
        self.psi0_gauss_y0_entry.grid(row=2, column=1)

        self.psi0_thomas_fermi_frame.grid_remove()
        self.psi0_gauss_frame.grid_remove()

        self.psi0_thomas_fermi_frame.grid()

        # Potential V
        self.V_option_list = ('Harmonisch', 'Harmonisch + Quartisch', 'Harmonisch + Optisch')
        self.V_option_sv = tk.StringVar(value=self.V_option_list[0])
        self.V_option_sv.trace_add("write", self.changeFrameV)

        V_label = tk.Label(self.top_right, text='Potential V')
        self.V_optionsmenu = tk.OptionMenu(self.top_right, self.V_option_sv, *self.V_option_list)

        V_label.grid(row=3, column=0)
        self.V_optionsmenu.grid(row=3, column=1, columnspan=4)

        ## refresh button
        self.V_refresh_button = tk.Button(self.top_right, command=self.updatePlot, text='Refresh')
        self.V_refresh_button.grid(row=5, column=5)

        # # V harmonic
        self.V_harmonic_frame = tk.Frame(self.top_right, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.V_harmonic_frame.grid(row=4, column=0, columnspan=5, padx=self.padx, pady=self.pady)

        V_harmonic_gamma_label = tk.Label(self.V_harmonic_frame, text='Gamma y')
        self.V_harmonic_gamma_sv = tk.StringVar(value=self.paramObj.potential_parameters["gamma_y"])
        self.V_harmonic_gamma_entry = tk.Entry(self.V_harmonic_frame, width=10, justify=tk.RIGHT, textvariable=self.V_harmonic_gamma_sv)
        self.V_harmonic_gamma_entry.bind("<FocusOut>", self.focusOut)
        self.V_harmonic_gamma_entry.bind("<Return>", self.focusOut)

        V_harmonic_gamma_label.grid(row=0, column=0)
        self.V_harmonic_gamma_entry.grid(row=0, column=1)

        # # harmonic quartic
        self.V_harmonic_quartic_frame = tk.Frame(self.top_right, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.V_harmonic_quartic_frame.grid(row=4, column=0, columnspan=5, padx=self.padx, pady=self.pady)

        V_harmonic_quartic_alpha_label = tk.Label(self.V_harmonic_quartic_frame, text='Alpha')
        self.V_harmonic_quartic_alpha_sv = tk.StringVar(value=self.paramObj.potential_parameters["alpha"])
        self.V_harmonic_quartic_alpha_entry = tk.Entry(self.V_harmonic_quartic_frame, width=10, justify=tk.RIGHT, textvariable=self.V_harmonic_quartic_alpha_sv)
        self.V_harmonic_quartic_alpha_entry.bind("<FocusOut>", self.focusOut)
        self.V_harmonic_quartic_alpha_entry.bind("<Return>", self.focusOut)

        V_harmonic_quartic_kappa_label = tk.Label(self.V_harmonic_quartic_frame, text='Kappa')
        self.V_harmonic_quartic_kappa_sv = tk.StringVar(value=self.paramObj.potential_parameters["kappa_quartic"])
        self.V_harmonic_quartic_kappa_entry = tk.Entry(self.V_harmonic_quartic_frame, width=10, justify=tk.RIGHT, textvariable=self.V_harmonic_quartic_kappa_sv)
        self.V_harmonic_quartic_kappa_entry.bind("<FocusOut>", self.focusOut)
        self.V_harmonic_quartic_kappa_entry.bind("<Return>", self.focusOut)

        V_harmonic_quartic_alpha_label.grid(row=0, column=0)
        self.V_harmonic_quartic_alpha_entry.grid(row=0, column=1)
        
        V_harmonic_quartic_kappa_label.grid(row=1, column=0)
        self.V_harmonic_quartic_kappa_entry.grid(row=1, column=1)

        # # harmonic optic
        self.V_harmonic_optic_frame = tk.Frame(self.top_right, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.V_harmonic_optic_frame.grid(row=4, column=0, columnspan=5, padx=self.padx, pady=self.pady)

        V_harmonic_optic_v0_label = tk.Label(self.V_harmonic_optic_frame, text='V0')
        self.V_harmonic_optic_v0_sv = tk.StringVar(value=self.paramObj.potential_parameters["V0"])
        self.V_harmonic_optic_v0_entry = tk.Entry(self.V_harmonic_optic_frame, width=10, justify=tk.RIGHT, textvariable=self.V_harmonic_optic_v0_sv)
        self.V_harmonic_optic_v0_entry.bind("<FocusOut>", self.focusOut)
        self.V_harmonic_optic_v0_entry.bind("<Return>", self.focusOut)

        V_harmonic_optic_kappa_label = tk.Label(self.V_harmonic_optic_frame, text='Kappa')
        self.V_harmonic_optic_kappa_sv = tk.StringVar(value=self.paramObj.potential_parameters["kappa_optic"])
        self.V_harmonic_optic_kappa_entry = tk.Entry(self.V_harmonic_optic_frame, width=10, justify=tk.RIGHT, textvariable=self.V_harmonic_optic_kappa_sv)
        self.V_harmonic_optic_kappa_entry.bind("<FocusOut>", self.focusOut)
        self.V_harmonic_optic_kappa_entry.bind("<Return>", self.focusOut)

        V_harmonic_optic_v0_label.grid(row=0, column=0)
        self.V_harmonic_optic_v0_entry.grid(row=0, column=1)
        
        V_harmonic_optic_kappa_label.grid(row=1, column=0)
        self.V_harmonic_optic_kappa_entry.grid(row=1, column=1)

        self.V_harmonic_frame.grid_remove()
        self.V_harmonic_optic_frame.grid_remove()
        self.V_harmonic_quartic_frame.grid_remove()

        self.V_harmonic_frame.grid()

    def init_bottom_left(self):
        dt_label = tk.Label(self.bottom_left, text='Delta t')
        self.dt_sv = tk.StringVar(value=self.paramObj.dt)
        self.dt_sv.trace_add("write", self.onChange)
        self.dt_entry = tk.Entry(self.bottom_left, width=10, justify=tk.RIGHT, textvariable=self.dt_sv)
        self.dt_entry.bind("<FocusOut>", self.focusOut)
        self.dt_entry.bind("<Return>", self.focusOut)
        self.dt_constraint_label = tk.Label(self.bottom_left, text='muss kleiner sein als {:1.2e}'.format(self.dt_constraint))

        epsilon_label = tk.Label(self.bottom_left, text='Epsilon Limit')
        self.epsilon_sv = tk.StringVar(value=self.paramObj.epsilon_limit)
        self.epsilon_sv.trace_add("write", self.onChange)
        self.epsilon_entry = tk.Entry(self.bottom_left, width=10, justify=tk.RIGHT, textvariable=self.epsilon_sv)
        self.epsilon_entry.bind("<FocusOut>", self.focusOut)
        self.epsilon_entry.bind("<Return>", self.focusOut)

        maxIter_label = tk.Label(self.bottom_left, text='maximale Iterationen')
        self.maxIter_sv = tk.StringVar(value=self.paramObj.maxIterations)
        self.maxIter_sv.trace_add("write", self.onChange)
        self.maxIter_entry = tk.Entry(self.bottom_left, width=10, justify=tk.RIGHT, textvariable=self.maxIter_sv)
        self.maxIter_entry.bind("<FocusOut>", self.focusOut)
        self.maxIter_entry.bind("<Return>", self.focusOut)

        filename_label = tk.Label(self.bottom_left, text='Dateiname')
        self.filename_sv = tk.StringVar(value=self.paramObj.filename)
        self.filename_entry = tk.Entry(self.bottom_left, width=30, justify=tk.RIGHT, textvariable=self.filename_sv)
        self.filename_entry.bind("<FocusOut>", self.focusOut)
        self.filename_entry.bind("<Return>", self.focusOut)

        self.filename_button = tk.Button(self.bottom_left, text='Ändern', command=self.chooseFile)

        epsilon_threshold_label = tk.Label(self.bottom_left, text='Epsilon Threshold')
        self.epsilon_threshold_sv = tk.StringVar(value=self.paramObj.epsilon_threshold)
        self.epsilon_threshold_sv.trace_add("write", self.onChange)
        self.epsilon_threshold_entry = tk.Entry(self.bottom_left, width=10, justify=tk.RIGHT, textvariable=self.epsilon_threshold_sv)
        self.epsilon_threshold_entry.bind("<FocusOut>", self.focusOut)
        self.epsilon_threshold_entry.bind("<Return>", self.focusOut)

        dt_label.grid(row=0, column=0)
        self.dt_entry.grid(row=0, column=1)
        self.dt_constraint_label.grid(row=0, column=2)

        epsilon_label.grid(row=1, column=0)
        self.epsilon_entry.grid(row=1, column=1)

        maxIter_label.grid(row=2, column=0)
        self.maxIter_entry.grid(row=2, column=1)

        filename_label.grid(row=3, column=0)
        self.filename_entry.grid(row=3, column=1, columnspan=2)
        self.filename_button.grid(row=3, column=3)

        epsilon_threshold_label.grid(row=4, column=0)
        self.epsilon_threshold_entry.grid(row=4, column=1)

        ## start button
        font_large = font.Font(family='Helvetica', size=25, weight='normal')
        self.start_button = tk.Button(self.bottom_left, command=self.startCalculation, text='Starte Berechnung', width=15, height=1, font=font_large, bg='red')
        self.start_button.grid(row=5, column=0, columnspan=5)

    def init_bottom_right(self):
        fig = Figure(figsize=(5, 4), dpi=100)
        # fig.tight_layout()
        # plt.close('all')
        ax = fig.add_subplot(111)
        self.im = ax.imshow(self.paramObj.V, cmap='jet')
        self.cb = fig.colorbar(self.im)

        self.V_canvas = FigureCanvasTkAgg(fig, master=self.bottom_right)  # A tk.DrawingArea.
        self.V_canvas.draw()
        self.V_canvas.get_tk_widget().grid(row=0, column=0)

    def changeFramePsi0(self, a, b, c):
        choice = self.psi0_option_sv.get()
        if choice == "Thomas-Fermi-Approximation":
            self.psi0_gauss_frame.grid_remove()
            self.psi0_thomas_fermi_frame.grid()
            self.paramObj.psi0_choice = Psi0Choice.THOMAS_FERMI
        elif choice == "Gauss":
            self.psi0_thomas_fermi_frame.grid_remove()
            self.psi0_gauss_frame.grid()
            self.paramObj.psi0_choice = Psi0Choice.GAUSS
        else:
            messagebox.showerror("ERROR", "Choice for Psi0 not recognized, may not be implemented...")
            self.paramObj.choice_psi0 = Psi0Choice.NOT_IMPLEMENTED
        if self.applyChanges():
            self.paramObj.initV()
            self.updatePlot()
            self.update_dt_constraint()

    def changeFrameV(self, a, b, c):
        choice = self.V_option_sv.get()
        if choice == "Harmonisch":
            self.V_harmonic_optic_frame.grid_remove()
            self.V_harmonic_quartic_frame.grid_remove()
            self.V_harmonic_frame.grid()
            self.paramObj.potential_choice = PotentialChoice.HARMONIC
        elif choice == "Harmonisch + Quartisch":
            self.V_harmonic_frame.grid_remove()
            self.V_harmonic_optic_frame.grid_remove()
            self.V_harmonic_quartic_frame.grid()
            self.paramObj.potential_choice = PotentialChoice.HARMONIC_QUARTIC
        elif choice == "Harmonisch + Optisch":
            self.V_harmonic_quartic_frame.grid_remove()
            self.V_harmonic_frame.grid_remove()
            self.V_harmonic_optic_frame.grid()
            self.paramObj.potential_choice = PotentialChoice.HARMONIC_OPTIC
        else:
            messagebox.showerror("ERROR", "Choice for potential V not recognized, may not be implemented...")
            self.paramObj.potential_choice = PotentialChoice.NOT_IMPLEMENTED

        if self.applyChanges():
            self.paramObj.initV()
            self.updatePlot()
            self.update_dt_constraint()

    def applyChanges(self):
        # top left
        try:
            self.paramObj.x_low = float(self.x_low_sv.get())
            self.paramObj.x_high = float(self.x_high_sv.get())
            self.paramObj.y_low = float(self.y_low_sv.get())
            self.paramObj.y_high = float(self.y_high_sv.get())
        except ValueError:
            messagebox.showerror("Werte-Fehler", "Werte für die Raster-Grenzen müssen Zahlen sein!")
            return False

        try:
            # add check for negative values
            self.paramObj.resolutionX = int(self.res_x_sv.get())
            self.paramObj.resolutionY = int(self.res_y_sv.get())
        except ValueError:
            messagebox.showerror("Werte-Fehler", "Werte für die Auflösung müssen ganze Zahlen sein!")
            return False

        # bottom left
        try:
            self.paramObj.dt = float(self.dt_sv.get())
        except ValueError:
            messagebox.showerror("Werte-Fehler", "Wert für den Zeitschritt muss ein Zahl sein!")
            return False
        
        
        try:
            self.paramObj.epsilon_limit = float(self.epsilon_sv.get())
        except ValueError:
            messagebox.showerror("Werte-Fehler", "Wert für Konvergenzparameter Epsilon muss eine Zahl sein!")
            return False

        try:
            self.paramObj.maxIterations = int(self.maxIter_sv.get())
        except ValueError:
            messagebox.showerror("Werte-Fehler", "Wert für die maximale Anzahl an Iterationen muss eine ganze Zahl sein!")
            return False
    
        # add check for filename
        self.paramObj.filename = self.filename_sv.get()

        try:
            self.paramObj.epsilon_threshold = float(self.epsilon_threshold_sv.get())
        except ValueError:
            messagebox.showerror("Werte-Fehler", "Wert für Epsilon Threshold muss eine Zahl sein!")
            return False

        # top right
        try:
            self.paramObj.omega = float(self.omega_sv.get())
        except ValueError:
            messagebox.showerror("Werte-Fehler", "Wert für Rotationsfrequenz Omega muss eine Zahl sein!")
            return False
        
        try:
            self.paramObj.beta2 = float(self.beta_sv.get())
        except ValueError:
            messagebox.showerror("Werte-Fehler", "Wert für Beta muss eine Zahl sein!")
            return False

        # # Psi 0
        if self.paramObj.psi0_choice == Psi0Choice.THOMAS_FERMI:
            try:
                self.paramObj.psi0_parameters['gamma_y'] = float(self.psi0_thomas_fermi_gamma_sv.get())
            except ValueError:
                messagebox.showerror("Werte-Fehler", "Wert für Gamma y muss eine Zahl sein!")
                return False
        elif self.paramObj.psi0_choice == Psi0Choice.GAUSS:
            try:
                self.paramObj.psi0_parameters['sigma'] = float(self.psi0_gauss_sigma_sv.get())
                self.paramObj.psi0_parameters['x0'] = float(self.psi0_gauss_y0_sv.get())
                self.paramObj.psi0_parameters['y0'] = float(self.psi0_gauss_y0_sv.get())
            except ValueError:
                messagebox.showerror("Werte-Fehler", "Alle Parameter für Psi0 müssen Zahlen sein!")
                return False

        # # potential V
        if self.paramObj.potential_choice == PotentialChoice.HARMONIC:
            try:
                self.paramObj.potential_parameters["gamma_y"] = float(self.V_harmonic_gamma_sv.get())
            except ValueError:
                messagebox.showerror("Werte-Fehler", "Wert für Gamma y muss eine Zahl sein!")
                return False
        elif self.paramObj.potential_choice == PotentialChoice.HARMONIC_QUARTIC:
            try:
                self.paramObj.potential_parameters["alpha"] = float(self.V_harmonic_quartic_alpha_sv.get())
                self.paramObj.potential_parameters["kappa_quartic"] = float(self.V_harmonic_quartic_kappa_sv.get())
            except ValueError:
                messagebox.showerror("Werte-Fehler", "Alle Parameter für V Harmonisch + Quartisch müssen Zahlen sein!")
                return False
        elif self.paramObj.potential_choice == PotentialChoice.HARMONIC_OPTIC:
            try:
                self.paramObj.potential_parameters["V0"] = float(self.V_harmonic_optic_v0_sv.get())
                self.paramObj.potential_parameters["kappa_optic"] = float(self.V_harmonic_optic_kappa_sv.get())
            except ValueError:
                messagebox.showerror("Werte-Fehler", "Alle Parameter für V Harmonisch + Optisch müssen Zahlen sein!")
                return False
        else:
            messagebox.showerror("ERROR", "Potential not recognized")
            return False

        return True

    def chooseFile(self):
        filetypes = [('hdf5 database files', '*.hdf5'), ('All files', '*')]
        filename = asksaveasfilename(title='Save as..', defaultextension='.hdf5', filetypes=filetypes)
        if filename:
            self.filename_sv.set(filename)

    def updatePlot(self):
        # print(self.paramObj)
        self.im.set_data(self.paramObj.V)
        self.cb.set_clim(np.min(self.paramObj.V), np.max(self.paramObj.V))
        self.cb.set_ticks(np.linspace(np.min(self.paramObj.V), np.max(self.paramObj.V), 6))
        self.cb.draw_all()
        self.V_canvas.draw()

    def onChange(self, a, b, c):
        pass
        
    def calc_dt_constraint(self):
        w = WaveFunction2D(self.paramObj)
        w.initPsi_0()
        b_ = self.paramObj.V + self.paramObj.beta2*np.abs(w.psi_array)**2
        bmin = np.min(b_)
        bmax = np.max(b_)
        self.dt_constraint = 2/(bmax+bmin)

    def update_dt_constraint(self):
        self.calc_dt_constraint()
        self.dt_constraint_label['text'] = 'muss kleiner sein als {:1.2e}'.format(self.dt_constraint)

    def focusOut(self, a):
        if self.applyChanges():
            self.paramObj.initV()
            self.updatePlot()
        self.update_dt_constraint()


    def startCalculation(self):
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
    root = tk.Tk()
    ParameterApp(root)
    root.mainloop()