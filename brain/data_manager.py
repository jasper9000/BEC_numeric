import numpy as np
import h5py
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .wave_function import WaveFunction2D
from .parameter_object import ParameterObject

class DataManager:
    def __init__(self, filename="default.hdf5", compression="gzip"):
        self.filename = filename
        self.file = None
        self.fileOpen = False
        self.compressionAlgo = compression
        self.incKey = 0

    def __del__(self):
        self.closeFile()
        print("[INFO] inside destructor of DataManager")

    def setGlobalAttributes(self, attributes):
        for key in attributes:
            self.file.attrs[key] = attributes[key]
    
    def newFile(self):
        if self.file or self.fileOpen:
            self.closeFile()
            raise Exception("A File was already open! Closed it to be sure...")
        self.file = h5py.File(self.filename, "w")
        if self.file:
            print("[INFO] Created new file", self.filename)
        self.fileOpen = True

    def loadFile(self):
        if self.file or self.fileOpen:
            self.closeFile()
            raise Exception("A File was already open! Closed it to be sure...")
        self.file = h5py.File(self.filename, "r+")
        if self.file:
            print("[INFO] Loaded file", self.filename)
        self.fileOpen = True
        self.incKey = self.getNFrames() + 1

    def closeFile(self):
        if not self.file or not self.fileOpen:
            print("[INFO] No open file present, no need to close anything.")
            return
        self.file.close()
        self.fileOpen = False

    def addDset(self, array, attributes):
        # make a new key
        key = "{:05d}".format(self.incKey)
        self.incKey += 1
        # save array
        self.file.create_dataset(key, array.shape, dtype=array.dtype, data=array, compression=self.compressionAlgo)
        for a in attributes:
            self.file[key].attrs[a] = attributes[a]

    def getDset(self, key):
        return np.array(self.file[key]), self.file[key].attrs

    def removeDset(self, key):
        del self.file[key]
    
    def numberDsets(self):
        return self.incKey+1

    def makeKey(self, key):
        return "{:05d}".format(key)
    
    def getNFrames(self):
        return len(self.file.keys())

    def getLastKey(self):
        return list(self.file.keys())[-1]

    def listInfo(self):
        print("File Attributes")
        for attr in self.file.attrs:
            print("{:20s} : {}".format(attr, self.file.attrs[attr]))
        print("{:20s} : {}".format('Number of Frames', self.getNFrames()))
        
    def returnLastFrame(self):
        frame, _ = self.getDset(self.getLastKey())
        return frame

    def listFrames(self):
        #each dataset is a frame, so each key is a frame
        frames = []
        for key in self.file.keys():
            frame, _ = self.getDset(key)
            frames.append(frame)
        return frames

    def saveFrames(self, filename='default.mp4', fps=10, dpi=500, dynamic_colorbar=True):
        print("Saving data to .mp4 file...")
        fig, ax = plt.subplots()
        shape = (self.file.attrs['resX'], self.file.attrs['resY'])
        data = np.random.rand(shape[0],shape[1])
        im = ax.imshow(data, cmap='jet', animated=True, vmin=0, vmax=1)
        fig.colorbar(im)
        tx = ax.set_title('')

        lastKey = int(self.getLastKey())
        endingFrameDelay = 3

        def update(i):
            if i > lastKey:
                key = "{:05d}".format(lastKey)
            else:
                key = "{:05d}".format(i)
            psi_array, attributes = self.getDset(key)
            im.set_data(np.abs(psi_array)**2)
            tx.set_text("Frame {}\nn = {}\nt = {:2.3f}".format(i, attributes['n'],  attributes['t']))
            if dynamic_colorbar:
                vmin = np.min(np.abs(psi_array)**2)
                vmax = np.max(np.abs(psi_array)**2)
                im.set_clim(vmin,vmax)
                print("Processing frame {} of {}, {:3.1f}% done".format(i, lastKey+endingFrameDelay*fps, 100*i/(lastKey+endingFrameDelay*fps)))
            return im
        anim = animation.FuncAnimation(fig, update, blit=False, frames=lastKey+endingFrameDelay*fps, interval=100)
        anim.save(filename, writer ='ffmpeg', fps=fps, dpi=dpi)

    def displayLastFrame(self, figsize=(10, 8)):
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

    def calcObservables(self):
        print("[INFO] Calculating Observables now, this may take a while.")
        file_attr = self.file.attrs
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
        for i in range(self.getNFrames()):
            array, _ = self.getDset(self.makeKey(i))
            w.setPsi(array)
            w.calcFFT()
            w.calcL()
            w.calcNabla()

            self.file[self.makeKey(i)].attrs['E'] = w.calcEnergy()
            self.file[self.makeKey(i)].attrs['L'] = w.calcL_expectation()
            self.file[self.makeKey(i)].attrs['Nabla'] = w.calcNabla_expectation()

    def getObservables(self):
        E, L, Nabla, t = [], [], [], []
        for i in range(self.getNFrames()):
            _, attributes = self.getDset(self.makeKey(i))
            t.append(attributes['t'])
            E.append(attributes['E'])
            L.append(attributes['L'])
            Nabla.append(attributes['Nabla'])
        return E, L, Nabla, t

    def plotObservables(self, calc=True):
        if calc:
            self.calcObservables()
        L = []
        E = []
        Nabla = []
        t = []
        for i in range(self.getNFrames()):
            _, attr = self.getDset(self.makeKey(i))
            t.append(attr['t'])

            Nabla.append(attr['Nabla'])
            E.append(attr['E'])
            L.append(attr['L'])
        
        plt.plot(t, E, label='E')
        plt.plot(t, L, label='L')
        plt.plot(t, Nabla, label='Nabla')
        plt.legend()
        plt.show()

    def displayFrames(self, playback_speed=20, dynamic_colorbar=True, figsize=(10, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        shape = (self.file.attrs['resX'], self.file.attrs['resY'])
        data = np.random.rand(*shape)
        im = ax.imshow(data, cmap='jet', animated=True, vmin=0, vmax=1)
        fig.colorbar(im)
        # tx = ax.set_title('Title', animated=True)
        tx = ax.text(0.05, 0.9, '', transform=ax.transAxes, color="w")
        # tx = ax.text(.5, 1.005, 'test', transform = ax.transAxes)

        fps = 1000//playback_speed
        lastKey = int(self.getLastKey())
        endingFrameDelay = 3

        def init():
            tx.set_text("init")
            im.set_data(np.zeros(shape))
            return tx, im

        def update(i):
            if i > lastKey:
                key = "{:05d}".format(lastKey)
            else:
                key = "{:05d}".format(i)
            psi_array, attributes = self.getDset(key)

            tx.set_text("Frame {} of {}\nn = {}\nt = {:2.3f}".format(i, lastKey+endingFrameDelay*fps, attributes['n'],  attributes['t']))
            im.set_data(np.abs(psi_array)**2)
            
            if dynamic_colorbar:
                vmin = np.min(np.abs(psi_array)**2)
                vmax = np.max(np.abs(psi_array)**2)
                im.set_clim(vmin,vmax)
            return tx, im
        anim = animation.FuncAnimation(fig, update, init_func=init, blit=True, frames=lastKey+endingFrameDelay*fps, interval=playback_speed)
        plt.show()
        return anim

    def displayFull(self, playback_speed=20, dynamic_colorbar=True, figsize=(10, 6)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        shape = (self.file.attrs['resX'], self.file.attrs['resY'])
        data = np.random.rand(*shape)
        im = ax1.imshow(data, cmap='jet', animated=True, vmin=0, vmax=1)
        cb = fig.colorbar(im, ax=ax1)
        # tx = ax.set_title('Title', animated=True)
        tx = ax1.text(0.05, 0.9, '', transform=ax1.transAxes, color="w")
        # tx = ax.text(.5, 1.005, 'test', transform = ax.transAxes)

        E, L, Nabla, t = self.getObservables()

        pl1, = ax2.plot(t, E, label="E")
        pl2, = ax2.plot(t, L, label="L")
        pl3, = ax2.plot(t, Nabla, label="Nabla")

        y_min = np.min((np.min(E), np.min(L), np.min(Nabla)))
        y_max = np.max((np.max(E), np.max(L), np.max(Nabla)))

        _, attributes_last_frame = self.getDset(self.getLastKey())
        ax2.set_xlim(0, attributes_last_frame['t'])
        ax2.set_ylim(y_min-10, y_max+10)
        ax2.legend()

        fps = 1000//playback_speed
        lastKey = int(self.getLastKey())
        endingFrameDelay = 3

        def update(i):
            if i > lastKey:
                key = "{:05d}".format(lastKey)
            else:
                key = "{:05d}".format(i)
            psi_array, attributes = self.getDset(key)

            tx.set_text("Frame {} of {}\nn = {}\nt = {:2.3f}".format(i, lastKey+endingFrameDelay*fps, attributes['n'],  attributes['t']))
            im.set_data(np.abs(psi_array)**2)
            
            if dynamic_colorbar:
                vmin = np.min(np.abs(psi_array)**2)
                vmax = np.max(np.abs(psi_array)**2)
                im.set_clim(vmin,vmax)

            # observables#
            if i > lastKey:
                a = lastKey
            else:
                a = i
            pl1.set_data(t[:a], E[:a])
            pl2.set_data(t[:a], L[:a])
            pl3.set_data(t[:a], Nabla[:a])

            cb.draw_all()
            
            return tx, im, pl1, pl2, pl3, cb

        anim = animation.FuncAnimation(fig, update, blit=True, frames=lastKey+endingFrameDelay*fps, interval=playback_speed)
        plt.show()
        return anim