import numpy as np
import pandas as pd
import vedo as v
import matplotlib.pyplot as plt
from sys import exit
from glob import glob
from copy import copy
import argparse
from datetime import date


today = str(date.today())  # "2020-08-03"
units = "pct"
unitsList = ["pct", "mm"]  # ["pct"] #["pct", "mm"]
debug = False

lobeCol = [
  (0.47843137254901963, 0.47843137254901963, 0.47843137254901963),
  (1.0, 170.0 / 255.0, 0.0),
  (1.0, 0.3333333333333333, 0.4980392156862745),
  (0.0, 0.6666666666666666, 1.0),
  (100.0 / 255.0, 220 / 255.0, 184.0 / 255.0),
]

# lungCol = ['blue', 'red']
lungCol = [lobeCol[1], lobeCol[-2]]

lobeIDs = {
  "RUL": "4",
  "RML": "5",
  "RLL": "6",
  "LUL": "7",
  "LLL": "8",
}
lobes = ["RUL", "RML", "RLL", "LUL", "LLL"]
# = ["blue","black", "green"]
lobeLine = ["-", "--", "-.", "-", "-."]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--caseID', '-c',
                    default='3948', 
                    type=str,#, required=True,
                    help='training data case'
                    )
parser.add_argument('--filename', '-f',
                    default='reconstruction', 
                    type=str,
                    help='string for filename [default is reconstruction]'
                    )
parser.add_argument('--out_dir', '-o',
                    default="morphologicalAnalysis/", 
                    type=str,
                    help='directory to find output volumes'
                    )
parser.add_argument('--gt_dir', '-g',
                    default="/home/josh/3DSlicer/project/luna16Rescaled/",
                    type=str,
                    help='directory to find ground truth surfaces'
                    )
args = parser.parse_args()


def get_vol(pred, gt):
  v_p = pred.volume()
  v_gt = gt.volume()
  return v_p, v_gt


def get_vol_error(v_p, v_gt):
  # print("predicted volume is", v_p)
  # print("gt volume is", v_gt)
  # print(abs(v_p-v_gt)/max(v_gt,v_p)*100,"%\n")
  return (v_p - v_gt) / (v_gt) * 100


print("Reading data")
surfName = "{}_case{}_{}.stl"

volKeys = lobes + ["Right"] + ["Left"] + ["Total"]
vol_error_keys = ["Right", "Left", "Total"]

gt_dir = f"{args.gt_dir}/case{args.caseID}/"
gt_name = args.caseID + "_mm_"  # "seg-case20-"
vol_error = dict.fromkeys(volKeys)
vol_pred = dict.fromkeys(volKeys)
vol_gt = dict.fromkeys(volKeys)

vol_pred["Right"] = 0.0
vol_pred["Left"] = 0.0
vol_pred["Total"] = 0.0
vol_gt["Right"] = 0.0
vol_gt["Left"] = 0.0
vol_gt["Total"] = 0.0

for key in volKeys:
  if key in lobes:
    surfDir = "surfaces/"
    vol_gt[key] = v.load(gt_dir + gt_name + lobeIDs[key] + ".stl").volume()
    vol_pred[key] = 0
    vol_error[key] = np.inf

    if key[0] == "R":
      vol_gt["Right"] += vol_gt[key]
    elif key[0] == "L":
      vol_gt["Left"] += vol_gt[key]
    vol_gt["Total"] += vol_gt[key]

vol_pred["Left"] = v.load(surfDir + surfName.format(args.filename, args.caseID, "left")).volume()
vol_pred["Right"] = v.load(surfDir + surfName.format(args.filename, args.caseID, "right")).volume()
vol_pred["Total"] = vol_pred["Left"] + vol_pred["Right"]
for key in vol_error_keys:
  vol_error[key] = (vol_pred[key] - vol_gt[key]) / vol_gt[key] * 100.

# write data like [shape, ground truth, reconstructon, diff [mm^3], diff [%]]
shape_names = list(vol_error.keys())
vol_pred_arr = np.round(np.array(list(vol_pred.values())), 2)
vol_gt_arr = np.round(np.array(list(vol_gt.values())), 2)
vol_error_arr = np.round(np.array(list(vol_error.values())), 2)
data_to_write = np.c_[
  shape_names,
  vol_gt_arr,
  vol_pred_arr,
  vol_pred_arr - vol_gt_arr,
  vol_error_arr,
]
print("error is ")
print(vol_error_arr)

pd.DataFrame(data_to_write).to_csv(
  f"{args.out_dir}/lobeVolumeStats{args.caseID}_{args.filename}.txt",
  sep="\t",
  index=False,
  header=[
    "shape",
    "ground truth [mm^3]",
    "reconstruction [mm^3]",
    "difference [mm^3]",
    "difference [%]",
  ],
  float_format='%g'
)
