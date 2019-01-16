from brain import DataManager
import os

filename = "saved_simulations/01.hdf5"

d = DataManager(filename)
d.loadFile()
d.listInfo()
d.displayFrames()
# d.saveFrames("default.mp4")
d.closeFile()