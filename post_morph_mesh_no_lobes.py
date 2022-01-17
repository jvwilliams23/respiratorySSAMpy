"""
Post-processing script for morphing template mesh to fit pre-computed landmarks
"""

import argparse
import numpy as np 
from glob import glob
from sys import exit
import vedo as v

from morphAirwayTemplateMesh import MorphAirwayTemplateMesh
from morphTemplateMesh import MorphTemplateMesh as MorphLobarTemplateMesh
from respiratoryReconstructionSSAM import RespiratoryReconstructSSAM

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
parser.add_argument('--landmark_dir', '-ld',
                    default="allLandmarks/", 
                    type=str,
                    help='directory to find ground truth landmarks'
                    )
parser.add_argument('--out_lm_dir', '-o',
                    default="outputLandmarks/", 
                    type=str,
                    help='directory to find output landmarks from SSAM'
                    )
args = parser.parse_args()


if __name__ == "__main__":
  case = args.caseID
  out_lm_dir = args.out_lm_dir
  landmark_dir = args.landmark_dir
  print("reading from ", landmark_dir)
  surf_dir = "surfaces/"
  # hard-code templates landmarks. 
  # this is one that appears to adapt best to other cases
  template_dir = "templates/coarserTemplates/"
  # template_lmFile = f"{landmark_dir}/allLandmarks3948.csv"
  # template_airway_file = (
  #   "segmentations/template3948/smooth_newtemplate3948_mm.stl"
  # )
  # strings to iterate through for dictionaries of various shapes
  shapes = ["Airway", "RUL", "RML", "RLL", "LUL", "LLL"]
  lobes = ["RUL", "RML", "RLL", "LUL", "LLL"]
  # numbering for each lobe in file
  lNums = {"RUL": "4", "RML": "5", "RLL": "6", "LUL": "7", "LLL": "8"}

  # find which set of landmarks correspond to each part of lungs
  landmark_ordering = dict.fromkeys(shapes)
  landmark_ordering["SKELETON"] = np.loadtxt(
    landmark_dir + "landmarkIndexSkeleton.txt"
  ).astype(int)
  landmark_ordering["LUNGS"] = []
  landmark_ordering["right"] = []
  landmark_ordering["left"] = []
  for shape in shapes:
    landmark_ordering[shape] = np.loadtxt(
      landmark_dir + f"landmarkIndex{shape}.txt", dtype=int
    )
    if shape[0].upper() == "R":
      landmark_ordering["right"].append(landmark_ordering[shape])
    if shape[0].upper() == "L":
      landmark_ordering["left"].append(landmark_ordering[shape])
  landmark_ordering["right"] = np.hstack(landmark_ordering["right"])
  landmark_ordering["left"] = np.hstack(landmark_ordering["left"])

  # file name for target landmarks to morph mesh to, and name of mesh to write
  target_lm_file_name = f"{args.filename}_case{case}"
  target_lm_file = f"{out_lm_dir}/{target_lm_file_name}_ALL.csv"
  target_surf_name = f"{surf_dir}/{target_lm_file_name}_airway.stl"
  # get carina to align mesh and landmarks to same coord system 
  template_lmFileOrig = "landmarks/manual-jw/landmarks3948.csv"
  carinaTemplate = (
    np.loadtxt(
      template_lmFileOrig, skiprows=1, delimiter=",", usecols=[1, 2, 3]
    )[1]
    * -1
  )

  # load mesh and landmarks for template and target
  # lm_template = np.loadtxt(
  #   template_lmFile, skiprows=1, delimiter=",", usecols=[0, 1, 2]
  # )
  # template_airway_mesh = v.load(template_airway_file)
  # template_airway_mesh = template_airway_mesh.pos(carinaTemplate)
  target_lms = np.loadtxt(target_lm_file, delimiter=",", skiprows=1)

  # morph_airway = MorphAirwayTemplateMesh(
  #   lm_template[landmark_ordering["Airway"]],
  #   target_lms[landmark_ordering["Airway"]],
  #   template_airway_mesh,
  #   sigma=0.3,
  #   quiet=False,
  # )
  # morph_airway.mesh_target.write(target_surf_name)

  for lung in ["right", "left"]:
    print(f"making {lung} template mesh")
    # load data for morphing
    template_lmFile = f"{landmark_dir}/allLandmarks8684.csv"
    lm_template = np.loadtxt(
      template_lmFile, skiprows=1, delimiter=",", usecols=[0, 1, 2]
    )
    template_mesh_file_lung = f"templates/8684_mm_{lung}.stl"
    template_lung_mesh = v.load(template_mesh_file_lung)
    morph_lung = MorphLobarTemplateMesh(
      lm_template[landmark_ordering[lung]],
      target_lms[landmark_ordering[lung]],
      template_lung_mesh,
      sigma=0.3,
      quiet=True,
    )
    target_lung_surf_out = f"{surf_dir}/{target_lm_file_name}_{lung}.stl"
    morph_lung.mesh_target.write(target_lung_surf_out)
