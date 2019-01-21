from brain import DataManager
import os
import matplotlib.pyplot as plt
import numpy as np

filename = 'D:/bec_data/supercritical_hd_long.hdf5'
# filename = 'saved_simulations/03.hdf5'

d = DataManager(filename)
d.loadFile()
d.listInfo()

# print(list(d.file.keys()))

# d.plotObservables()
# d.displayFrames()
# d.calcObservables()
# d.displayFull()
# d.displayLastFrame()

d.saveFrames("D:/bec_data/supercritical_hd_long.mp4", fps=60)
d.closeFile()