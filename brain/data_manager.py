__author__ = "Jasper Riebesehl"
__version__ = "1.0"
__email__ = "jasper.riebesehl@physnet.uni-hamburg.de"

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .wave_function import WaveFunction2D
from .parameter_object import ParameterObject, Psi0Choice, PotentialChoice

class DataManager:
    '''This class handles anything related to the saving of simulated data to .hdf5 data files.

    Attributes:
        filename: A string that contains the name of the file to open or create.
        file: pyh5 File object, a handle for the file once it is opened.
        fileOpen: A boolean that states whether a file is opened. 
        compressionAlgo: A string that sets the compression algorithm for pyh5. Default is gzip.
        incKey: An integer that contains the current number for the next keyframe.
    '''
    def __init__(self, filename="default.hdf5"):
        '''This function initializes this class.'''
        self.filename = filename
        self.file = None
        self.fileOpen = False
        self.compressionAlgo = 'gzip'
        self.incKey = 0

    def __del__(self):
        '''As the destructor of this class this funtion is called when a instance of this class is deleted.
        Closes any open files if present.'''
        self.closeFile()
        print("[INFO] inside destructor of DataManager")

    def setGlobalAttributes(self, attributes):
        '''This function adds global attributes to the hdf5 file.
        
        Arguments:
            atributes: A dictionary whose keys and corresponding values are added to the file.
        '''
        for key in attributes:
            self.file.attrs[key] = attributes[key]
    
    def newFile(self):
        '''This function creates a new file with the given filename. WARNING, this function will overwrite any file that may be present with the same name.
        After the function call the file will be open and ready to be written in.
        '''
        if self.file or self.fileOpen:
            self.closeFile()
            raise Exception("A File was already open! Closed it to be sure...")
        self.file = h5py.File(self.filename, "w")
        if self.file:
            print("[INFO] Created new file", self.filename)
        self.fileOpen = True

    def loadFile(self):
        '''This function will open an already present file in read/write mode. This mode should NOT be used to add more frames to a file,
        but rather to calculate the observables and display the file.
        '''
        if self.file or self.fileOpen:
            self.closeFile()
            raise Exception("A File was already open! Closed it to be sure...")
        self.file = h5py.File(self.filename, "r+")
        if self.file:
            print("[INFO] Loaded file", self.filename)
        self.fileOpen = True
        self.incKey = self.getNFrames() + 1

    def closeFile(self):
        '''This function closes any opened files. This function should always be called after the file is no longer used.
        '''
        if not self.file or not self.fileOpen:
            print("[INFO] No open file present, no need to close anything.")
            return
        self.file.close()
        self.fileOpen = False

    def addDset(self, array, attributes):
        '''This function adds a frame of simulation and adds it to the file. This should be used to add a frame to the file.

        Arguments:
            array: A 2D numpy array that may contain complex values.
            attributes: A dictionary whose contents will be added to the dataset as attributes. 
        '''
        # make a new key
        key = "{:05d}".format(self.incKey)
        self.incKey += 1
        # save array
        self.file.create_dataset(key, array.shape, dtype=array.dtype, data=array, compression=self.compressionAlgo)
        for a in attributes:
            self.file[key].attrs[a] = attributes[a]

    def getDset(self, key):
        '''Returns a dataset and its attributes for a given key.
        Arguments:
            key: A string that has to be the key for a dataset in the file.
        
        Returns:
            A tupel with 2 elements. The first entry is the array itself and
            the second entry is a dictionary that contains the attributes of that frame.
            (array, attributes)'''
        return np.array(self.file[key]), self.file[key].attrs

    def removeDset(self, key):
        '''Deletes a dataset given a valid key.'''
        del self.file[key]
    
    def numberDsets(self):
        '''Returns the number of datasets in a file.'''
        return self.incKey+1

    def makeKey(self, key):
        '''Creates a formated key for easy use.
        
        Argument:
            key: An integer, high than the current number of data sets in the file.
            
        Return:
            A string that is 5 characters long and is basically the integer with leading zeros.
        '''
        return "{:05d}".format(key)
    
    def getNFrames(self):
        '''Returns the number of datasets in a file.
        '''
        return len(self.file.keys())

    def getLastKey(self):
        '''Returns the key of the frame with the highest integer key, which should be the last frame.
        '''
        return list(self.file.keys())[-1]

    def listInfo(self):
        '''A function that prints out all global attributes, of the file.
        '''
        print("File Attributes")
        for attr in self.file.attrs:
            print("{:30s} : {}".format(attr, self.file.attrs[attr]))

    def returnInfo(self):
        '''Returns a string that contains all global attributes of the file.
        '''
        s = ''
        for attr in self.file.attrs:
            if attr == "potential_choice":
                s += "{:25s} : {}\n".format(attr, PotentialChoice(int(self.file.attrs[attr])).__str__())
            elif attr == "psi0_choice":
                s += "{:25s} : {}\n".format(attr, Psi0Choice(int(self.file.attrs[attr])).__str__())
            else:
                s += "{:25s} : {}\n".format(attr, self.file.attrs[attr])
        return s
        
    def returnLastFrame(self):
        '''Returns the array of the dataset with the highest integer key, which should be the last frame.
        '''
        frame, _ = self.getDset(self.getLastKey())
        return frame

    def listFrames(self):
        '''This functions returns a list that contains the arrays of each dataset.
        '''
        #each dataset is a frame, so each key is a frame
        frames = []
        for key in self.file.keys():
            frame, _ = self.getDset(key)
            frames.append(frame)
        return frames

    def saveFrames(self, filename='default.mp4', fps=10, dpi=500, dynamic_colorbar=True):
        '''This function saves the sequence of datasets to a .mp4 video file. This will only work if ffmpeg is installed on the system.

        Arguments:
            filename: A string that contains the name of the .mp4 file.
            fps: An integer that determines the number of frames per second of video.
            dpi: An integer that determines the resolution of the video.
            dynamic_colorbar: A boolean that determines whether the range of the colorbar changes to have the optimal display range.
        '''
        print("[INFO] Saving data to .mp4 file...")
        fig, ax = plt.subplots()
        shape = (self.file.attrs['resX'], self.file.attrs['resY'])
        data = np.random.rand(shape[0],shape[1])
        im = ax.imshow(data, cmap='jet', animated=True, vmin=0, vmax=1)
        fig.colorbar(im)
        tx = ax.set_title('')

        # set up a delay to the end of the video to display the last frame longer.
        lastKey = int(self.getLastKey())
        endingFrameDelay = 3

        def update(i):
            # This function gets called every time a new frame is rendered.
            if i > lastKey:
                key = "{:05d}".format(lastKey)
            else:
                key = "{:05d}".format(i)
            # get the next array to display
            psi_array, attributes = self.getDset(key)
            im.set_data(np.abs(psi_array)**2)
            tx.set_text("Frame {}\nn = {}\nt = {:2.3f}".format(i, attributes['n'],  attributes['t']))
            if dynamic_colorbar:
                # change the range of the colobar
                vmin = np.min(np.abs(psi_array)**2)
                vmax = np.max(np.abs(psi_array)**2)
                im.set_clim(vmin,vmax)
                print("[INFO] Processing frame {} of {}, {:3.1f}% done".format(i, lastKey+endingFrameDelay*fps, 100*i/(lastKey+endingFrameDelay*fps)))
            return im
        # set up animation renderer
        anim = animation.FuncAnimation(fig, update, blit=False, frames=lastKey+endingFrameDelay*fps, interval=100)
        anim.save(filename, writer ='ffmpeg', fps=fps, dpi=dpi)

    def displayLastFrame(self, figsize=(10, 8)):
        '''This function produces a plot that shows the last frame of the simulation, which should be the ground state.
        '''
        frame, attributes = self.getDset(self.getLastKey())
        frame = np.abs(frame)**2
        vmin = np.min(frame)
        vmax = np.max(frame) 
        plt.figure(figsize=(10, 8))
        plt.imshow(frame, cmap='jet', vmin=vmin, vmax=vmax)
        plt.colorbar()
        text = "n = {}\nt = {:2.3f}".format(attributes['n'],  attributes['t'])
        plt.text(0.05, 0.9, text, color="w")
        plt.show()

    def are_observables_calculated(self):
        '''This function checks whether the observables of every frame are already calculates and returns the corresponding boolean value.
        '''
        file_attr = self.file.attrs
        try:
            _ = file_attr['calculated_observables']
        except:
            self.file.attrs['calculated_observables'] = False
        return file_attr['calculated_observables']

    def calcObservables(self):
        '''This function checks if the observables have to be calculated and does so if they are not.
        '''
        file_attr = self.file.attrs
        try:
            _ = file_attr['calculated_observables']
        except:
            self.file.attrs['calculated_observables'] = False

        # check whether the observables have to be calculated. 
        if file_attr['calculated_observables']:
            print("[INFO] The Observables were already calculated.")
        else:
            # they have to be calculated.
            # create a parameter object from the global attributes.
            print("[INFO] Calculating Observables now, this may take a while.")
            V_param = {
                "gamma_y" : file_attr['potential_gamma_y'],
                "alpha" : file_attr['potential_alpha'],
                "V0" : file_attr['potential_V0'],
                "kappa_optic" : file_attr['potential_kappa_optic'],
                "kappa_quartic" : file_attr['potential_kappa_quartic']
            }
            p = ParameterObject(resolutionX=file_attr['resX'], resolutionY=file_attr['resY'],
            x_low=file_attr['x_low'], x_high=file_attr['x_high'], y_low=file_attr['y_low'], y_high=file_attr['y_high'],
            beta2=file_attr['beta'], omega=file_attr['omega'], potential_choice=file_attr['potential_choice'],
            potential_parameters=V_param)

            p.initV()
            w = WaveFunction2D(p)

            # calculate the observables for every frame
            for i in range(self.getNFrames()):
                # get dataset and calculate precursor components
                array, _ = self.getDset(self.makeKey(i))
                w.setPsi(array)
                w.calcFFT()
                w.calcL()
                w.calcNabla()

                # calculate and add observables as attributes to the dataset (frame)
                self.file[self.makeKey(i)].attrs['E'] = w.calcEnergy()
                self.file[self.makeKey(i)].attrs['L'] = w.calcL_expectation()
                self.file[self.makeKey(i)].attrs['Nabla'] = w.calcNabla_expectation()
            self.file.attrs['calculated_observables'] = True
            print("[INFO] Calculated the obsevables successfully.")

    def getObservables(self):
        '''This function returns list for every observable for every frame.

        Returns:
            4 lists with length equal to the number of frames which contain the observable value for every frame.
            The order of the lists is

            E, L, Nabla, t

            where
            E: energy
            L: angular momentum
            Nabla: kinetic energy
            t: time of the step
        '''
        self.calcObservables()
        E, L, Nabla, t = [], [], [], []
        for i in range(self.getNFrames()):
            attributes = self.file[self.makeKey(i)].attrs
            t.append(attributes['t'])
            E.append(attributes['E'])
            L.append(attributes['L'])
            Nabla.append(attributes['Nabla'])
        return E, L, Nabla, t

    def plotObservables(self):
        '''Produces a plot that shows the observables over time.
        '''
        E, L, Nabla, t = self.getObservables()
        
        plt.plot(t, E, label='E')
        plt.plot(t, L, label='L')
        plt.plot(t, Nabla, label='Nabla')
        plt.legend()
        plt.show()

    def displayFrames(self, playback_speed=20, dynamic_colorbar=True, figsize=(10, 8)):
        '''Produces an animated display of the simulation.

        Arguments:
            playback_speed: An integer that determines the delay inbetween two frames.
            dynamic_colorbar: A boolean that determines whether the range of the colorbar changes dynamically.
            figsize: A 2-tuple of integers that determines the size of the display.
        '''
        # set up the matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        shape = (self.file.attrs['resX'], self.file.attrs['resY'])
        data = np.random.rand(*shape)
        im = ax.imshow(data, cmap='jet', animated=True, vmin=0, vmax=1)
        fig.colorbar(im)
        tx = ax.text(0.05, 0.9, '', transform=ax.transAxes, color="w")

        # set up addition of additional ending frames
        fps = 1000//playback_speed
        lastKey = int(self.getLastKey())
        endingFrameDelay = 3

        def init():
            # this function is pointless but i'm scared to remove it...
            tx.set_text("init")
            im.set_data(np.zeros(shape))
            return tx, im

        def update(i):
            # this function is called on each new frame
            if i > lastKey:
                key = "{:05d}".format(lastKey)
            else:
                key = "{:05d}".format(i)
            psi_array, attributes = self.getDset(key)

            tx.set_text("Frame {} of {}\nn = {}\nt = {:2.3f}".format(i, lastKey+endingFrameDelay*fps, attributes['n'],  attributes['t']))
            # get the data for the next frame
            im.set_data(np.abs(psi_array)**2)
            
            if dynamic_colorbar:
                # update colorbar range
                vmin = np.min(np.abs(psi_array)**2)
                vmax = np.max(np.abs(psi_array)**2)
                im.set_clim(vmin,vmax)
            return tx, im
        # set up matplotlib animation
        anim = animation.FuncAnimation(fig, update, init_func=init, blit=True, frames=lastKey+endingFrameDelay*fps, interval=playback_speed)
        plt.show()
        return anim

    def displayFull(self, playback_speed=20, dynamic_colorbar=True, figsize=(10, 6)):
        '''Produces an animated display of the simulation. Additionally displays the observables over time.

        Arguments:
            playback_speed: An integer that determines the delay inbetween two frames.
            dynamic_colorbar: A boolean that determines whether the range of the colorbar changes dynamically.
            figsize: A 2-tuple of integers that determines the size of the display.
        '''
        # set up matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        shape = (self.file.attrs['resX'], self.file.attrs['resY'])
        data = np.random.rand(*shape)
        im = ax1.imshow(data, cmap='jet', animated=True, vmin=0, vmax=1)
        fig.colorbar(im, ax=ax1)
        tx = ax1.text(0.05, 0.9, '', transform=ax1.transAxes, color="w")

        # get observables to display
        E, L, Nabla, t = self.getObservables()

        # set up observable plot
        pl1, = ax2.plot(t, E, label="E")
        pl2, = ax2.plot(t, L, label="L")
        pl3, = ax2.plot(t, Nabla, label="Nabla")

        y_min = np.min((np.min(E), np.min(L), np.min(Nabla)))
        y_max = np.max((np.max(E), np.max(L), np.max(Nabla)))

        _, attributes_last_frame = self.getDset(self.getLastKey())
        ax2.set_xlim(0, attributes_last_frame['t'])
        ax2.set_ylim(y_min-10, y_max+10)
        ax2.legend()

    
        # set up additional ending frames
        fps = 1000//playback_speed
        lastKey = int(self.getLastKey())
        endingFrameDelay = 3

        def update(i):
            # this function produces the next frame of the animation
            if i > lastKey:
                key = "{:05d}".format(lastKey)
            else:
                key = "{:05d}".format(i)
            psi_array, attributes = self.getDset(key)

            tx.set_text("Frame {} of {}\nn = {}\nt = {:2.3f}".format(i, lastKey+endingFrameDelay*fps, attributes['n'],  attributes['t']))
            # get data for next frame
            im.set_data(np.abs(psi_array)**2)
            
            if dynamic_colorbar:
                # update colorbar range
                vmin = np.min(np.abs(psi_array)**2)
                vmax = np.max(np.abs(psi_array)**2)
                im.set_clim(vmin,vmax)

            # update observable range to display
            if i > lastKey:
                a = lastKey
            else:
                a = i
            pl1.set_data(t[:a], E[:a])
            pl2.set_data(t[:a], L[:a])
            pl3.set_data(t[:a], Nabla[:a])
            
            return tx, im, pl1, pl2, pl3

        # set up matplotlib animation
        anim = animation.FuncAnimation(fig, update, blit=True, frames=lastKey+endingFrameDelay*fps, interval=playback_speed)
        plt.show()
        return anim