'''
evaluate airway reconstruction by comparing landmark to landmark distance.
'''

import numpy as np
import pandas as pd 
import userUtils as utils
from glob import glob
import argparse
from sys import exit


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case', '-c',
                    default='8684', 
                    type=str,#, required=True,
                    help='training data case'
                    )
parser.add_argument('-f', '--filename',
                    type=str, 
                    default="landmarkDistanceStatsparameterTest1_kd{}kr{}cg{}ca{}_case{}_{}.csv.txt",
                    help='filename to read'
                    )
args = parser.parse_args()
caseID = args.case
file_name = args.filename

paramVals = []

out_dir = 'morphologicalAnalysis/'

anatom_list="2.0 1.0 0.6 0.2 0.05".split()
grad_list="1.0 0.2 0.02 0.0".split()
radius_list="7 14 21 28".split()
dist_list="10 15 20 25 30 35".split()

results = [None]*len(dist_list)
all_results = []
all_params = []

for i, kd in enumerate(dist_list):
 print(kd)
 results[i] = [None]*len(radius_list)
 for j, kr in enumerate(radius_list):
  results[i][j] = [None]*len(grad_list)
  if kr > kd:
   continue
  for k, cg in enumerate(grad_list):
   results[i][j][k] = [None]*len(anatom_list)
   for l, ca in enumerate(anatom_list):
    file_name_args = file_name.format(kd, kr, cg, ca, '*', 'ALL')
    results[i][j][k][l] = [None]*len(glob(out_dir+file_name_args))
    res_split = []
    raw_results = []
    for f, filei in enumerate(glob(out_dir+file_name_args)):
     #print(filei)
     results[i][j][k][l][f] = np.loadtxt(filei)
     raw_results.extend(list(results[i][j][k][l][f]))
     #res_split.extend([results[i][j][k][l][f].mean(), np.percentile(results[i][j][k][l][f], 0.95), results[i][j][k][l][f].max()])
    #all_results.append([results[i][j][k][l].mean(), np.percentile(results[i][j][k][l], 0.95), results[i][j][k][l].max()])
    '''if len(res_split) != 9:
     print('skipping')
     continue
    print(res_split)
    all_results.append(res_split)'''
    all_results.append([np.mean(raw_results), np.percentile(raw_results, 95), np.max(raw_results)])
    all_params.append([kd, kr, cg, ca])

print(np.array(all_results).min(axis=0))
min_loc = np.array(all_results).argmin(axis=0)
print(min_loc)
for loc in min_loc:
 print('\nbest hyperparameters')
 print(np.array(all_params)[loc])
 print('results', all_results[loc])
