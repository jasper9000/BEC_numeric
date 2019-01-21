import tkinter as tk
from tkinter import filedialog

import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np
from wave_function import ParameterObject

class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # set up standard parameters
        self.paramObj = ParameterObject(resolutionX=256, resolutionY=256,
        x_low=-16, x_high=16, y_low=-16, y_high=16,
        beta2=1000, omega=0.9)

        self.padx = 5
        self.pady = 5
        # self.init_window()
        self.init_frames()
        self.init_top_left()
        self.init_top_right()
        self.init_bottom_left()
        self.init_bottom_right()

    def init_frames(self):
        self.top_left = tk.LabelFrame(self.parent, text='Raster-Parameter', labelanchor='nw', width=400, height=400, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.top_right = tk.LabelFrame(self.parent, text='Physikalische Parameter', labelanchor='nw', width=400, height=400, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.bottom_left = tk.LabelFrame(self.parent, text='Numerische Parameter', labelanchor='nw', width=400, height=400, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.bottom_right = tk.LabelFrame(self.parent, text='Potential V', labelanchor='nw', width=400, height=400, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        
        # layout all of the main containers
        # self.parent.grid_rowconfigure(1, weight=1)
        # self.parent.grid_columnconfigure(0, weight=1)

        self.top_left.grid(row=0, column=0, padx=self.padx, pady=self.pady)
        self.top_right.grid(row=0, column=1, padx=self.padx, pady=self.pady)
        self.bottom_left.grid(row=1, column=0, padx=self.padx, pady=self.pady)
        self.bottom_right.grid(row=1, column=1, padx=self.padx, pady=self.pady)

    def init_top_left(self):
        # grid spacing parameters
        x_low_label = tk.Label(self.top_left, text='x low')
        self.x_low_entry = tk.Entry(self.top_left, width=10, justify=tk.RIGHT)

        x_high_label = tk.Label(self.top_left, text='x high')
        self.x_high_entry = tk.Entry(self.top_left, width=10, justify=tk.RIGHT)

        y_low_label = tk.Label(self.top_left, text='y low')
        self.y_low_entry = tk.Entry(self.top_left, width=10, justify=tk.RIGHT)

        y_high_label = tk.Label(self.top_left, text='y high')
        self.y_high_entry = tk.Entry(self.top_left, width=10, justify=tk.RIGHT)

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
        self.res_x_entry = tk.Entry(self.resolution_frame, width=10, justify=tk.RIGHT)

        res_y_label = tk.Label(self.resolution_frame, text='Auflösung y-Richtung', bg=color_red)
        self.res_y_entry = tk.Entry(self.resolution_frame, width=10, justify=tk.RIGHT)

        res_x_label.grid(row=0, column=0)
        self.res_x_entry.grid(row=0, column=1)

        res_y_label.grid(row=1, column=0)
        self.res_y_entry.grid(row=1, column=1)

    def init_top_right(self):
        # omega and beta
        omega_label = tk.Label(self.top_right, text='Omega')
        self.omega_entry = tk.Entry(self.top_right, width=10, justify=tk.RIGHT)

        beta_label = tk.Label(self.top_right, text='Beta')
        self.beta_entry = tk.Entry(self.top_right, width=10, justify=tk.RIGHT)

        omega_label.grid(row=0, column=0)
        self.omega_entry.grid(row=0, column=1)

        beta_label.grid(row=0, column=3)
        self.beta_entry.grid(row=0, column=4)

        # Psi 0
        self.psi0_frame = tk.Frame(self.top_right, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.psi0_frame.grid(row=1, column=0, columnspan=5, padx=self.padx, pady=self.pady)

        psi0_option_list = ('Thomas-Fermi-Approximation', 'Gauß')
        self.psi0_option = tk.StringVar()
        self.psi0_option.set(psi0_option_list[0])

        psi0_label = tk.Label(self.psi0_frame, text='Start-Wellenfunktion')
        self.psi0_optionsmenu = tk.OptionMenu(self.psi0_frame, self.psi0_option, *psi0_option_list)

        psi0_label.grid(row=0, column=0)
        self.psi0_optionsmenu.grid(row=0, column=1)

        # Potential V
        self.V_frame = tk.Frame(self.top_right, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.V_frame.grid(row=2, column=0, columnspan=5, padx=self.padx, pady=self.pady)

        V_option_list = ('Harmonisch', 'Harmonisch + Quartisch', 'Harmonisch + Optisch')
        self.V_option = tk.StringVar()
        self.V_option.set(V_option_list[0])

        V_label = tk.Label(self.V_frame, text='Potential V')
        self.V_optionsmenu = tk.OptionMenu(self.V_frame, self.V_option, *V_option_list)

        V_label.grid(row=0, column=0)
        self.V_optionsmenu.grid(row=0, column=1)

    def init_bottom_left(self):
        dt_label = tk.Label(self.bottom_left, text='Delta t')
        self.dt_entry = tk.Entry(self.bottom_left, width=10, justify=tk.RIGHT)

        epsilon_label = tk.Label(self.bottom_left, text='Epsilon')
        self.epsilon_entry = tk.Entry(self.bottom_left, width=10, justify=tk.RIGHT)

        maxIter_label = tk.Label(self.bottom_left, text='maximale Iterationen')
        self.maxIter_entry = tk.Entry(self.bottom_left, width=10, justify=tk.RIGHT)

        filename_label = tk.Label(self.bottom_left, text='Dateiname')
        self.filename_entry = tk.Entry(self.bottom_left, width=30, justify=tk.RIGHT)

        dt_label.grid(row=0, column=0)
        self.dt_entry.grid(row=0, column=1)

        epsilon_label.grid(row=1, column=0)
        self.epsilon_entry.grid(row=1, column=1)

        maxIter_label.grid(row=1, column=3)
        self.maxIter_entry.grid(row=1, column=4)

        filename_label.grid(row=2, column=0)
        self.filename_entry.grid(row=2, column=1, columnspan=3)

    def init_bottom_right(self):
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        x = np.linspace(-16, 16, 100)
        xx, yy = np.meshgrid(x, x)
        im = ax.imshow(np.exp(np.sin(xx+yy)), cmap='jet')
        fig.colorbar(im)

        canvas = FigureCanvasTkAgg(fig, master=self.bottom_right)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().grid()


    def init_window(self):
        tk.Label(self.parent, text="Beta").grid(row=0, padx=self.padx, pady=self.pady)
        tk.Label(self.parent, text="Omega").grid(row=1, padx=self.padx, pady=self.pady)

        self.e1 = tk.Entry(self.parent)
        self.e2 = tk.Entry(self.parent)

        self.e1.grid(row=0, column=1, padx=self.padx, pady=self.pady)
        self.e2.grid(row=1, column=1, padx=self.padx, pady=self.pady)

    def read_values(self):
        pass

       
if __name__ == "__main__":
    print("main")
    root = tk.Tk()
    MainApplication(root)#.pack(side="top", fill="both", expand=True)
    root.mainloop()