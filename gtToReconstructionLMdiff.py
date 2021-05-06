'''
evaluate airway reconstruction by comparing landmark to landmark distance.
'''

import numpy as np
import pandas as pd 
import vedo as v
import userUtils as utils
import matplotlib.pyplot as plt
from glob import glob
import argparse


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case', '-c',
                    default=False, 
                    type=str,#, required=True,
                    help='training data case'
                    )
args = parser.parse_args()
caseID = args.case


# caseID = '3948'

lm_index_file = 'allLandmarks/landmarkIndexAirway.txt'
lm_index = np.loadtxt(lm_index_file).astype(int)
out_dir = 'outputLandmarks/reconstruction{}_{}.csv'
gt_dir = 'allLandmarks/allLandmarks{}.csv'
gt_mesh_dir = "/home/josh/project/imaging/airwaySSAMpy/segmentations/airwaysForRadiologistWSurf/{}/*.stl" 

out_lm = np.loadtxt(out_dir.format(caseID, 'ALL'), delimiter=',', skiprows=1)[lm_index]
gt_lm = np.loadtxt(gt_dir.format(caseID), delimiter=',', skiprows=1)[lm_index]

# center landmarks
out_lm -= out_lm.mean(axis=0)
offset_file = 'landmarks/manual-jw/landmarks{}.csv'.format(caseID)
carina = np.loadtxt(offset_file, skiprows=1, delimiter=',', usecols=[1,2,3])[1]
gt_offset =  carina + gt_lm.mean(axis=0)
gt_lm -= gt_lm.mean(axis=0)
gt_mesh = v.load(glob(gt_mesh_dir.format(caseID))[0]).pos(-gt_offset)
# gt_mesh.alignTo(v.Points(gt_lm), rigid=True)

dist = utils.euclideanDist(out_lm, gt_lm)
print('max dist is', dist.max())
print('distance statistics are')
print(pd.DataFrame(dist).describe())
# gt_pts = v.Points(gt_lm, r=4).c('black')
out_pts = v.Points(out_lm, r=4).cmap('hot', dist)
outliers = v.Points(out_lm[dist>10], r=10).c('black')
v.show(gt_mesh.alpha(0.2), out_pts, outliers)
plt.hist(dist, bins=100)
plt.xlabel('landmark-to-landmark distance [mm]')
plt.show()

print('TODO - get curvature per branch!')
