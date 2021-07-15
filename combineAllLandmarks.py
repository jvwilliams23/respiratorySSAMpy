'''
Take landmarked lobes and airways. Apply some saved transformations to them
such that they are in the same coordinate frame.
'''
import numpy as np 
from glob import glob 
import matplotlib.pyplot as plt
import vedo as v

# save global paths to landmarks
airway_dir = '/home/josh/project/imaging/airwaySSAMpy/landmarks/manual-jw-diameterFromSurface/'
lobe_dir = '/home/josh/project/imaging/lobarSSMpy/master/gamesData_coarse/'
# transform for airways (carina is at index 1 landmark in array)
airway_dir_orig = '/home/josh/project/imaging/airwaySSAMpy/landmarks/manual-jw/'
# transformation for lobes
trans_dir = "/home/josh/project/imaging/lobarSSMpy/master/savedPointClouds/allLandmarks/"
# where to write combined landmarks to
out_dir = 'allLandmarks/'

lobe_list = ['RUL', 'RML', 'RLL', 'LUL', 'LLL']


airway_files = glob(airway_dir+'*.csv')
airway_files_orig = glob(airway_dir_orig+'landmarks*.csv') # to get offset
airway_files.sort()
airway_files_orig.sort()
# take all files for random lobe so we can get the patient IDs.
# assumes that lobe_dir is not missing landmarks for any patients!
base_lobe_files = glob(lobe_dir+'landmarks*RUL*.csv')
assert len(base_lobe_files) == 51, 'some files are missing. Need to adjust code'

# get IDs of patients from landmarks. sort landmark files to be correctly ordered
base_lobe_files = sorted(base_lobe_files, 
                      key=lambda x: int(x.replace(".csv","")[-4:]))
patientIDs = [i.split("/")[-1].replace(".csv", "")[-4:] for i in base_lobe_files]
lmIDs = [i.split("/")[-1].split("landmarks")[1][:4] for i in airway_files]

# load landmarks into np array
airway_lms =  np.array([np.loadtxt(l, delimiter=",",skiprows=1) 
                            for l in airway_files])
airway_lms_orig = np.array([np.loadtxt(l, delimiter=",",skiprows=1,usecols=[1,2,3]) 
                            for l in airway_files_orig])
carinaArr = airway_lms_orig[:,1]
airway_lms = airway_lms + carinaArr[:,np.newaxis] # offset airway back to CT space

# filter out patients that are in lobe list but not airways
delInd = []
for i, currID in enumerate(patientIDs):
  if currID not in lmIDs:
    delInd.append(i)
for dId in delInd[::-1]:
  patientIDs.pop(dId)

# initialise lists and dicts for storing data
landmarks = airway_lms.copy()
lobe_lms = dict.fromkeys(lobe_list)
lmTrans_all = dict.fromkeys(lobe_list)
cols = ['green', 'black', 'red', 'pink', 'salmon']
# plt.close()
vp = v.Plotter()
lobeInds = []
lmCounter = 0
# output array index for each shape's landmarks
inds = np.arange(0,landmarks.shape[1])+lmCounter
lmCounter += inds.size
np.savetxt(out_dir+'landmarkIndex{}.txt'.format('Airway'), inds, fmt="%i")
for i, (lobe, c) in enumerate(zip(lobe_list, cols)):
  print('reading {}'.format(lobe))
  # read and sort lobe landmarks
  lobe_files = glob( lobe_dir+"/landmarks*{0}*.csv".format(lobe) )
  lobe_files = sorted(lobe_files, 
                      key=lambda x: int(x.replace(".csv","")[-4:]))
  # read and sort lobe transformations
  transDirs_all = glob(trans_dir+"transformParams_case*_m_{}.dat".format(lobe))
  transDirs_all = sorted(transDirs_all, 
                        key=lambda x: int(x.split("case")[1][:4]))
  # filter out patients missing from airway datasets
  for dId in delInd[::-1]:
    lobe_files.pop(dId)
    transDirs_all.pop(dId)
  assert len(transDirs_all) == len(lobe_files), 'mismatching number of files'

  # load to numpy and apply offset
  lobe_lms[lobe] =  np.array([np.loadtxt(l, delimiter=",") 
                                  for l in lobe_files])
  lmTrans_all[lobe] = np.vstack([np.loadtxt(t, skiprows=1, max_rows=1) 
                                    for t in transDirs_all])
  lobe_lms[lobe] = lobe_lms[lobe] + lmTrans_all[lobe][:,np.newaxis]
  # vp += v.Points(lobe_lms[lobe][0], r=3).c(c)
  inds = np.arange(0,lobe_lms[lobe].shape[1])+lmCounter
  lmCounter += inds.size
  np.savetxt(out_dir+'landmarkIndex{}.txt'.format(lobe), inds, fmt="%i")
  landmarks = np.hstack((landmarks, lobe_lms[lobe]))

landmarks -= carinaArr[:,np.newaxis]
for lm, pId in zip(landmarks, patientIDs):
  out_file = out_dir+'allLandmarks{}.csv'.format(pId)
  np.savetxt(out_file, lm, header='x, y, z', delimiter=',')

#   plt.scatter(lobe_lms[lobe][0,:,0], lobe_lms[lobe][0,:,2], c=c)
# plt.scatter(airway_lms[0,:,0], airway_lms[0,:,2],c='blue')
# plt.show()
# vp += v.Points(landmarks[0], r=3).c('blue')
# vp.show()
