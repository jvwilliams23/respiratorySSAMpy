from sys import argv
import numpy as np

fname = argv[1]
data = np.loadtxt(fname, delimiter=",")
data[:, 1] /= 2.0
np.savetxt(fname, data, delimiter=",")
