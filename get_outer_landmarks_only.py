import argparse
import random
import re
from concurrent import futures
from copy import copy
from datetime import date
from distutils.util import strtobool
from glob import glob
from math import pi
from os import remove
from sys import argv, exit
from time import time

import hjson
import matplotlib.pyplot as plt
import networkx as nx
import nevergrad as ng
import numpy as np
import vedo as v
from scipy.spatial.distance import cdist, pdist
from skimage import io
from skimage.color import rgb2gray
from sklearn.preprocessing import StandardScaler
from vedo import printc

import userUtils as utils

from morphAirwayTemplateMesh import MorphAirwayTemplateMesh
from morphTemplateMesh import MorphTemplateMesh as MorphLobarTemplateMesh
from respiratoryReconstructionSSAM import RespiratoryReconstructSSAM

# from reconstructSSAM import LobarPSM
from respiratorySAM import RespiratorySAM
from respiratorySSAM import RespiratorySSAM
from respiratorySSM import RespiratorySSM


def filesFromRegex(path):
  """
  Given an input template path as BASEDIR/..../file-in-dir-.....ext
  Find all files that match the regex
  """
  regex = re.compile(path)
  all_paths = glob(path.replace(".", "*"))
  match_path = [p for p in all_paths if re.match(regex, p)]
  match_path.sort()
  return match_path


def matchesFromRegex(path):
  """
  Given an input template path as BASEDIR/..../file-in-dir-.....ext
  Find all strings that represented by regex wildcards i.e. if
  path = DRRs/..../drr-.....ext
  and
  match = DRRs/0596/drr-0596.ext
  add 0596 to a list
  Parameters
  ----------
  string of a path with wildcards to represent various case IDs
  """
  regex = re.compile(path)
  all_paths = glob(path.replace(".", "*"))
  matches = []
  for p in all_paths:
    if re.match(regex, p):
      id_match = ""
      for search_letter, match_letter in zip(path, p):
        if search_letter == "." and match_letter != ".":
          id_match += match_letter
          # print('match', match_letter, search_letter)
        elif search_letter != "." and id_match != "":
          break
      matches.append(id_match)
  matches.sort()
  return matches


def getFissureLandmarks(coords, adjacent_lobes):
  """
  Get landmarks representing fissure between lobes.
  Args:
      coords (dict): dict containing 2D array of coordinates for each lobe
  Returns:
      landmark_fissure_index (dict): contains IDs for each point on lobe fissure
  """
  landmark_fissure_index = dict.fromkeys(adjacent_lobes.keys())
  # -store initial distance between fissure landmarks

  for shape1 in adjacent_lobes.keys():
    landmark_fissure_index[shape1] = []

    for shape2 in adjacent_lobes[shape1]:
      dist_mah = cdist(coords[shape1], coords[shape2], metric="mahalanobis")
      # -cutoff = 0.05 for only edges, equal to ~0.15 - 0.2 for all
      # -fissure points (but also includes some others)
      cutoff = 0.2
      landmark_fissure_index[shape1] = landmark_fissure_index[shape1] + list(
        np.where(dist_mah < cutoff)[0]
      )

    landmark_fissure_index[shape1] = list(
      np.unique(landmark_fissure_index[shape1])
    )
  return landmark_fissure_index


if __name__ == "__main__":
  write_dir = "allLandmarks_noFissures/"

  with open("config_3proj.json") as f:
    config = hjson.load(f)
  # read landmark data
  landmarkDirs = filesFromRegex(config["luna16paths"]["landmarks"])
  lmIDs = matchesFromRegex(config["luna16paths"]["landmarks"])
  landmarkDirsOrig = glob("landmarks/manual-jw/landmarks*.csv")
  landmarkDirs.sort()
  landmarkDirsOrig.sort()

  shape_list = ["Airway", "RUL", "RML", "RLL", "LUL", "LLL"]
  lobes_list = ["RUL", "RML", "RLL", "LUL", "LLL"]
  # -dict of lobes and adjacent lobes (used for getting fissures)
  adjacent_lobes = {
    "RUL": ["RML", "RLL"],
    "RML": ["RUL", "RLL"],
    "RLL": ["RUL", "RML"],
    "LUL": ["LLL"],
    "LLL": ["LUL"],
  }

  landmarks = np.array(
    [np.loadtxt(l, delimiter=",", skiprows=1) for l in landmarkDirs]
  )

  # find which landmark indexes correspond to each shape in the lung
  lmOrder = dict.fromkeys(shape_list)
  lmOrder["LUNGS"] = []
  for shape in shape_list:
    lmOrder[shape] = np.loadtxt(
      f"allLandmarks/landmarkIndex{shape}.txt", dtype=int
    )
    if shape in lobes_list:
      lmOrder["LUNGS"].extend(list(lmOrder[shape]))
  lmOrder["LUNGS"] = np.array(lmOrder["LUNGS"])

  landmarks_hull_only = []
  for i, lm in enumerate(landmarks):
    input_file_name = landmarkDirs[i].split("/")[1]
    print(input_file_name)
    ordered_landmarks = dict.fromkeys(shape_list)
    for shape in shape_list:
      ordered_landmarks[shape] = copy(landmarks[i][lmOrder[shape]])

    # find landmarks on lobar fissures and save for use later.
    # deleting same LMs allows us to maintain correspondence
    if i == 0:
      fissure_ids = getFissureLandmarks(ordered_landmarks, adjacent_lobes)
      for shape in lobes_list:
        write_name = f"{write_dir}/fissure_landmark_ids_{shape}.txt"
        np.savetxt(write_name, fissure_ids[shape], fmt="%i")
    else:
      fissure_ids = dict.fromkeys(lobes_list)
      for shape in lobes_list:
        write_name = f"{write_dir}/fissure_landmark_ids_{shape}.txt"
        fissure_ids[shape] = np.loadtxt(write_name, dtype=int)

    no_fissure_landmarks = dict.fromkeys(shape_list)
    no_fissure_landmarks["ALL"] = []

    # format landmarks to remove fissures and keep outer lobar landmarks
    # as well as airways
    for shape in shape_list:
      if shape in lobes_list:
        no_fissure_landmarks[shape] = np.delete(
          ordered_landmarks[shape], fissure_ids[shape], axis=0
        )
      else:
        # should only be airways
        no_fissure_landmarks[shape] = copy(ordered_landmarks[shape])

      no_fissure_landmarks["ALL"].append(no_fissure_landmarks[shape])
    no_fissure_landmarks["ALL"] = np.vstack(no_fissure_landmarks["ALL"])
    # if i == 0:
    #   vp = v.Plotter()
    #   vp += v.Points(no_fissure_landmarks["ALL"], r=5)
    #   vp.show()
    #   exit()

    np.savetxt(
      f"{write_dir}/{input_file_name}",
      no_fissure_landmarks["ALL"],
      delimiter=",",
      header="x, y, z"
    )
    landmarks_hull_only.append(no_fissure_landmarks)

    # get new landmark index files
    if i == 0:
      lm_order_adjusted = dict.fromkeys(shape_list)
      lm_counter = 0
      for shape in shape_list:
        # cumulative ID for landmarks in present shape
        num_lms_in_shape = len(no_fissure_landmarks[shape])
        lm_order_adjusted[shape] = np.arange(
          lm_counter, lm_counter + num_lms_in_shape
        )
        lm_counter += num_lms_in_shape
        # save new order and update max num pts
        np.savetxt(
          f"{write_dir}/landmarkIndex{shape}.txt",
          lm_order_adjusted[shape],
          fmt="%i",
        )
