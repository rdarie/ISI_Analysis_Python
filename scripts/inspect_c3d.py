import btk

reader = btk.btkAcquisitionFileReader()

file_path = 'Z:\ISI\ISI-C\ISI-C 0002\GoldenCopy\Transfer3\Brown Data\ISI-C\ISI-C-0002\Day14_AM\Block0017.c3d'
reader.SetFilename(file_path) # set a filename to the reader
reader.Update()
acq = reader.GetOutput() # acq is the btk aquisition object