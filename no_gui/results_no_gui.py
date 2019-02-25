from brain import DataManager

# filename of the .hdf5 file generated by the simulation
filename = 'D:/bec_data/test_0.hdf5'

# set up data manager object to load the file
d = DataManager(filename)
d.loadFile()
d.listInfo()

# different functions to manually look at the contents of this file
# d.calcObservables()
# d.plotObservables()
# d.displayFull()

d.displayFrames()
# d.displayLastFrame()

# uncomment the following line if you want to save the animation in .mp4 format.
# this will only work if ffmpeg is installed on the system.
# d.saveFrames("D:/bec_data/supercritical_1.mp4", fps=45)

# always close the file at the end
d.closeFile()
