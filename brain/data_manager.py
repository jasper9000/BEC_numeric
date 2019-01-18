import numpy as np
import h5py
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

