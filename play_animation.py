from brain import DataManager
import os
import matplotlib.pyplot as plt
import numpy as np

filename = 'D:/bec_data/06.hdf5'
# filename = 'saved_simulations/03.hdf5'

d = DataManager(filename)
d.loadFile()
d.listInfo()

# print(list(d.file.keys()))

d.displayFrames()
# d.displayLastFrame()

# d.saveFrames("D:/bec_data/04_fps30.mp4", fps=30)
d.closeFile()