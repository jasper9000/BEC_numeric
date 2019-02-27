import tkinter as tk
import tkinter.font as font
from tkinter import messagebox
from tkinter.filedialog import asksaveasfilename, askopenfilename

import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
from .data_manager import DataManager

class ResultsApp(tk.Frame):
    '''This class produces the GUI in which the user can look at the results of a previous simulation.
    '''
    def __init__(self, parent, *args, **kwargs):
        # initializes the GUI
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # init a DataManager to load a file
        self.data_manager = DataManager()

        # set up control variables for animation
        self.animation_is_playing = False
        self.step_forward = False
        self.step_backward = False
        self.current_animation_frame = 0
        self.last_animation_frame = 0
        self.animation_delay = 1

        # set up lists for observables
        self.E = []
        self.Nabla = []
        self.L = []
        self.t = []

        self.padx = 5
        self.pady = 5

        # call init functions
        self.init_frames()
        self.init_top_left()
        self.init_top_right()
        self.init_bottom()

    def __del__(self):
        # destructor of the GUI, closes any files that may be opened.
        self.data_manager.__del__()

    def init_frames(self):
        # initializes frames of the GUI
        self.topLevelFrame = tk.Frame(self.parent)
        self.topLevelFrame.grid()

        # set up frames
        self.top_left = tk.Frame(self.topLevelFrame, width=300, height=300, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.top_right = tk.LabelFrame(self.topLevelFrame, text='File Parameters', labelanchor='nw', width=300, height=300, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)
        self.bottom = tk.Frame(self.topLevelFrame, width=300, height=300, padx=self.padx, pady=self.pady, relief=tk.GROOVE, borderwidth=3)

        # place frames on the window
        self.top_left.grid(row=0, column=0, columnspan=2, padx=self.padx, pady=self.pady, sticky='')
        self.top_right.grid(row=0, column=2, padx=self.padx, pady=self.pady, sticky='')
        self.bottom.grid(row=1, column=0, columnspan=3, padx=self.padx, pady=self.pady, sticky='')

    def init_top_left(self):
        # initializes the top left frame which contains all control buttons for the animation.
        padY = 20

        # filename control
        filename_label = tk.Label(self.top_left, text="Filename")
        self.filename_sv = tk.StringVar(value="")
        self.filename_entry = tk.Entry(self.top_left, width=45, justify=tk.RIGHT, textvariable=self.filename_sv, state=tk.DISABLED)

        # open/close file buttons
        self.open_file_button = tk.Button(self.top_left, command=self.openFile, text='Open File')
        self.close_file_button = tk.Button(self.top_left, command=self.closeFile, text='Close File')

        # save as button
        self.save_as_button = tk.Button(self.top_left, command=self.saveFrames, text='Save as .mp4 File')
        save_as_label = tk.Label(self.top_left, text="Converting to .mp4 format will only work if ffmpeg is installed on the system.\nThe conversion will freeze the GUI, just be patient and it will come back to life.")

        # animation control buttons
        self.play_button = tk.Button(self.top_left, command = self.continueAnimation, text='Play')
        self.pause_button = tk.Button(self.top_left, command = self.pauseAnimation, text='Pause')
        self.step_forward_button = tk.Button(self.top_left, command = self.stepForward, text='Step >>')
        self.step_backward_button = tk.Button(self.top_left, command = self.stepBackward, text='Step <<')

        skip_frames_label = tk.Label(self.top_left, text='Number of Frames skipped on each screen refresh.\nA high value increases playback speed.')
        self.skip_frames_iv = tk.IntVar(value=0)
        self.skip_frames_scale = tk.Scale(self.top_left, from_=0, to_=15, resolution=1, tickinterval=5, orient=tk.HORIZONTAL, variable=self.skip_frames_iv, length=200)

        # place all elements on the window
        filename_label.grid(row=0, column=0, columnspan=4, pady=5)
        self.filename_entry.grid(row=1, column=0, columnspan=5, pady=padY)
        self.open_file_button.grid(row=1, column=5, pady=padY)
        self.close_file_button.grid(row=1, column=6, pady=padY)

        self.save_as_button.grid(row=2, column=0, pady=padY)
        save_as_label.grid(row=2, column=1, columnspan=4)

        self.play_button.grid(row=3, column=0, pady=padY)
        self.pause_button.grid(row=3, column=1, pady=padY)
        self.step_backward_button.grid(row=3, column=2, pady=padY)
        self.step_forward_button.grid(row=3, column=3, pady=padY)

        skip_frames_label.grid(row=4, column=0, columnspan=4, pady=padY)
        self.skip_frames_scale.grid(row=4, column=4, columnspan=2, pady=padY)

    def init_top_right(self):
        # initializes the top right frame which contains a list of parameters of a loaded file.
        self.parameters_text = tk.Text(self.top_right, width=60, height=20)
        self.parameters_text.grid(row=0, column=0, sticky="nsew")

        scrollb = tk.Scrollbar(self.top_right, command=self.parameters_text.yview)
        scrollb.grid(row=0, column=1, sticky='nsew')
        self.parameters_text['yscrollcommand'] = scrollb.set

    def init_bottom(self):
        # initializes the bottom frame of the GUI which contains the matplotlib animation
        # set up figure
        self.fig, (self.ax0, self.ax1, self.ax2) = plt.subplots(1,3,figsize=(12,4))

        self.fig.tight_layout(pad=2, w_pad=2.5, h_pad=2)

        # set up axes labeling
        self.ax0.set_title("density probability distribution $|\\Psi|^2$")
        self.ax0.set_xlabel("$x$")
        self.ax0.set_xticks(np.linspace(0, 255, 5))
        self.ax0.set_xticklabels(np.linspace(-16, 16, 5))
        self.ax0.set_ylabel("$y$")
        self.ax0.set_yticks(np.linspace(0, 255, 5))
        self.ax0.set_yticklabels(np.linspace(-16, 16, 5))

        self.ax1.set_title("phase of $\\Psi$")
        self.ax1.set_xlabel("$x$")
        self.ax1.set_xticks(np.linspace(0, 255, 5))
        self.ax1.set_xticklabels(np.linspace(-16, 16, 5))
        self.ax1.set_ylabel("$y$")
        self.ax1.set_yticks(np.linspace(0, 255, 5))
        self.ax1.set_yticklabels(np.linspace(-16, 16, 5))

        self.ax2.set_title("observables of $\\Psi$")
        self.ax2.set_xlabel("Imaginary Time $t$")

        # set up subplots
        shape = (256, 256)
        data = np.random.rand(*shape)
        self.im0 = self.ax0.imshow(data, cmap='jet', animated=True, vmin=0, vmax=1)
        self.im1 = self.ax1.imshow(data, cmap='jet', animated=True, vmin=0, vmax=1)

        self.pl1, = self.ax2.plot([0], [0], label="Energy $E$")
        self.pl2, = self.ax2.plot([0], [0], label="Angular momentum $L$")
        self.pl3, = self.ax2.plot([0], [0], label="Kinetic Energy")

        self.ax2.legend(fontsize=9)

        # set up tkinter canvas to draw on
        self.V_canvas = FigureCanvasTkAgg(self.fig, master=self.bottom)
        self.V_canvas.draw()
        self.V_canvas.get_tk_widget().grid(row=0, column=0)

        # set up toolbar for zoom etc.
        self.V_toolbar_frame = tk.Frame(self.bottom)
        self.V_toolbar_frame.grid()
        self.V_toolbar = NavigationToolbar2Tk(self.V_canvas, self.V_toolbar_frame)
        self.V_toolbar.update()

    def openFile(self):
        # this function is called when the user presses the open file button.
        # Creates a dialog where the user can choose a file.
        filetypes = [('hdf5 database files', '*.hdf5'), ('All files', '*')]
        filename = askopenfilename(title='Open..', defaultextension='.hdf5', filetypes=filetypes)

        # if a file was selected:
        if filename:
            # update GUI
            self.filename_sv.set(filename)
            self.data_manager.filename = filename

            # Giving out information about loading the file...
            self.clearParameterWidget()
            self.parameters_text.insert(tk.END, '\nTrying to load the file now.\n')
            self.parameters_text.insert(tk.END, 'This might take a while, depending on the filesize.\n')
            try:
                # use DataManager to load the file
                self.parent.after(0, self.data_manager.loadFile())
                self.open_file_button['state'] = tk.DISABLED
                ## set up plots
                # set up axis labels for image plots
                x_high = self.data_manager.file.attrs['x_high']
                x_low = self.data_manager.file.attrs['x_low']
                y_high = self.data_manager.file.attrs['y_high']
                y_low = self.data_manager.file.attrs['y_low']

                self.ax0.set_xticklabels(np.linspace(x_low, x_high, 5))
                self.ax0.set_yticklabels(np.linspace(y_low, y_high, 5))

                self.ax1.set_xticklabels(np.linspace(x_low, x_high, 5))
                self.ax1.set_yticklabels(np.linspace(y_low, y_high, 5))

                # set up observable plot
                if not self.data_manager.are_observables_calculated():
                    self.parameters_text.insert(tk.END, '\nThe observables have not yet been calculated!\nThis will take a bit.\nPlease do not close the window, it possibly corrupts the file.\n')
                # calculates the observables.
                # this is what is freezing the GUI when loading a new file...
                self.E, self.L, self.Nabla, self.t = self.data_manager.getObservables()
                
                y_min = np.min((np.min(self.E), np.min(self.L), np.min(self.Nabla)))
                y_max = np.max((np.max(self.E), np.max(self.L), np.max(self.Nabla)))

                _, attributes_last_frame = self.data_manager.getDset(self.data_manager.getLastKey())
                self.ax2.set_xlim(0, attributes_last_frame['t'])
                self.ax2.set_ylim(y_min-10, y_max+10)

                self.last_animation_frame = int(self.data_manager.getLastKey())
                self.resetParameterWidget()

            except Exception as inst:
                # if an error occurs, just show it
                messagebox.showerror("ERROR: {}".format(type(inst)), inst)

            # start the animation
            self.initAnimation()
            self.animation_is_playing = True
        
    def closeFile(self):
        # this function is called when the user presses the close file button and it closes any open files
        self.animation_is_playing = False
        self.data_manager.closeFile()
        self.open_file_button['state'] = tk.NORMAL

    def saveFrames(self):
        # this function is called when the user presses the save as .mp4 file
        # and it tries to save the animation in mp4 format.
        # this function will only succeed if ffmpeg is installed on the system
        print("[INFO] Saving the animation in .mp4 format, this will take a while.")
        try:
            filetypes = [('mp4 video files', '*.mp4'), ('All files', '*')]
            f = asksaveasfilename(title='Save as..', defaultextension='.mp4', filetypes=filetypes)
            fps = 40
            if f:
                self.data_manager.saveFrames(f, fps=fps, dpi=500)
        except Exception as inst:
            messagebox.showerror("ERROR: {}".format(type(inst)), inst)

    def initAnimation(self):
        # kick-starts the animation
        self.current_animation_frame = 0
        self.animation = animation.FuncAnimation(self.fig, self.updateAnimation, interval=self.animation_delay, blit=True)
        self.V_canvas.draw()

    def updateAnimation(self, i):
        # this function is called every time a new frame of the animation has to be drawn.
        # check if any control buttons were pressed and react acordingly
        if self.animation_is_playing or self.step_forward or self.step_backward:
            if self.step_forward or self.animation_is_playing:
                self.current_animation_frame += 1 + self.skip_frames_iv.get()
            elif self.step_backward:
                self.current_animation_frame -= 1 + self.skip_frames_iv.get()

            # making sure we only get valid values for the frame index
            if self.current_animation_frame > self.last_animation_frame:
                self.current_animation_frame = 0
            elif self.current_animation_frame < 0:
                self.current_animation_frame = self.last_animation_frame
            
            # make the key for the data manager frame
            if self.current_animation_frame > self.last_animation_frame:
                key = self.data_manager.makeKey(self.last_animation_frame)
            else:
                key = self.data_manager.makeKey(self.current_animation_frame)
            psi_array, _ = self.data_manager.getDset(key)

            # update image plots
            self.im0.set_data(np.abs(psi_array)**2)
            self.im1.set_data(np.angle(psi_array))

            vmin = np.min(np.abs(psi_array)**2)
            vmax = np.max(np.abs(psi_array)**2)
            self.im0.set_clim(vmin,vmax)

            # observables#
            key = int(key)
            self.pl1.set_data(self.t[:key], self.E[:key])
            self.pl2.set_data(self.t[:key], self.L[:key])
            self.pl3.set_data(self.t[:key], self.Nabla[:key])

            self.step_backward = False
            self.step_forward = False

        return self.im0, self.im1, self.pl1, self.pl2, self.pl3

    def continueAnimation(self):
        # is called when button play is pressed
        self.animation_is_playing = True
    
    def pauseAnimation(self):
        # is called when button pause is pressed
        self.animation_is_playing = False

    def stepForward(self):
        # is called when button step >> is pressed
        self.step_forward = True

    def stepBackward(self):
        # is called when button step << is pressed
        self.step_backward = True

    def resetParameterWidget(self):
        # displays the file info on the big text widget
        self.parameters_text.delete("@0,0", tk.END)
        self.parameters_text.insert("@0,0", self.data_manager.returnInfo())

    def clearParameterWidget(self):
        # deletes any characters that are present on the big text widget
        self.parameters_text.delete("@0,0", tk.END)