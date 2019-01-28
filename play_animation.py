from brain import DataManager
import os

print(os.getcwd())

# filename = 'D:/bec_data/gamma.hdf5'
filename = 'D:/bec_data/supercritical_1.hdf5'

d = DataManager(filename)
d.loadFile()
d.listInfo()

# d.plotObservables()
# d.displayFrames()
d.calcObservables()
d.displayFull()
# d.displayLastFrame()

# d.saveFrames("D:/bec_data/gammfa_3.mp4", fps=40)
d.closeFile()