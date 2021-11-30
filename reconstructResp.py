"""
    run script for reconstructing airways amd lobes from an X-ray

    if no input is given, default will create random test set of size = 1
"""

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

""" TODO - sync SAM and SSM classes to conda folder. Can then oimport like
from ssm import SSM as RespiratorySSM 
"""
from morphAirwayTemplateMesh import MorphAirwayTemplateMesh
from morphTemplateMesh import MorphTemplateMesh as MorphLobarTemplateMesh
from respiratoryReconstructionSSAM import RespiratoryReconstructSSAM

# from reconstructSSAM import LobarPSM
from respiratorySAM import RespiratorySAM
from respiratorySSAM import RespiratorySSAM
from respiratorySSM import RespiratorySSM

tag = "_case0"
template_mesh = "segmentations/template3948/newtemplate3948_mm.stl"
plotAlignment = False


def getInputs():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--inp",
    "-i",
    default="allLandmarks/",
    type=str,
    help="input files (landmarks)",
  )
  parser.add_argument(
    "--load_data_only",
    default="False",
    type=strtobool,
    help="only load input data. Do not perform optimisation.",
  )
  parser.add_argument(
    "--case",
    "-c",
    default="none",  #'3948',
    type=str,  # , required=True,
    help="training data case",
  )
  parser.add_argument(
    "--out",
    "-o",
    default="reconstruction",
    type=str,
    help="output surface tag ",
  )
  parser.add_argument(
    "--var",
    "-v",
    default=0.9,
    type=float,
    help="fraction of variance in training set to include in model [0,1]",
  )
  parser.add_argument(
    "--c_prior",
    "-cp",
    default=0.025,
    type=float,
    help="prior shape loss coefficient",
  )
  parser.add_argument(
    "--c_dense", "-cd", default=1.0, type=float, help="density loss coefficient"
  )
  parser.add_argument(
    "--c_edge",
    "-ce",
    default=0.01,
    type=float,
    help="edge map loss coefficient",
  )
  parser.add_argument(
    "--c_anatomical",
    "-ca",
    default=0.2,
    type=float,
    help="anatomical shadow loss coefficient",
  )
  parser.add_argument(
    "--c_grad",
    "-cg",
    default=0.0,
    type=float,
    help="image gradient loss coefficient",
  )
  parser.add_argument(
    "--kernel_radius",
    "-kr",
    default=14,
    type=int,
    help="radius (pixels) of image kernels",
  )
  parser.add_argument(
    "--kernel_distance",
    "-kd",
    default=20,
    type=int,
    help="distance (pixels) between image kernels",
  )
  parser.add_argument(
    "--drrs",
    default="./DRRs/luna16/",
    # default="../xRaySegmentation/DRRs_enhanceAirway/luna16_cannyOutline/",
    type=str,
    help="input files (drr)",
  )
  parser.add_argument(
    "--meshdir",
    "-m",
    default="segmentations/",
    type=str,
    help="directory of surface files",
  )
  parser.add_argument(
    "--shapes",
    default="Airway RUL RML RLL LUL LLL",
    type=str,
    help="which shape would the user like to grow?"
    + "Corresponds to string common in landmarks text files"
    + "\nRUL, RML, RLL, LUL, LLL, or ALL?",
  )
  parser.add_argument(
    "--debug",
    "-d",
    default=False,
    type=bool,
    help="debug mode -- shows print checks and blocks" + "plotting outputs",
  )
  parser.add_argument(
    "--epochs",
    "-e",
    default=4000,
    type=int,
    help="number of optimisation iterations",
  )
  parser.add_argument(
    "--xray",
    "-x",
    default=False,
    type=str,  # required=True,
    help="X-ray outline to use for fitting (2xN csv)",
  )
  parser.add_argument(
    "--newMean",
    default=str(False),
    type=strtobool,
    help="Create new mean mesh [default false]",
  )
  parser.add_argument(
    "--newProjLM",
    default=str(False),
    type=strtobool,
    help="Create and save new projected landmarks [default false]",
  )
  parser.add_argument(
    "--imgSpacing",
    default=1,
    type=int,
    help="multiplier to coarsen images (must be int)",
  )
  parser.add_argument(
    "-r",
    "--randomise",
    default=str(False),
    type=strtobool,
    help="choose random test samples [default is false]",
  )
  parser.add_argument(
    "--keep_in_training",
    default=str(False),
    type=strtobool,
    help="If true, test data is kept in for training [default is false]",
  )
  parser.add_argument(
    "-ts",
    "--testSize",
    default=1,
    type=int,
    help="number of samples in test set [default is 1]",
  )
  parser.add_argument(
    "-pf",
    "--plot_freq",
    default=250,
    type=int,
    help="save figure showing SSM on XR after this many iterations",
  )
  parser.add_argument(
    "-q",
    "--quiet",
    default=str(False),
    type=strtobool,
    help="Turn off print checks",
  )
  parser.add_argument(
    "-b",
    "--bayesian_opt",
    default=str(False),
    type=strtobool,
    help="Perform bayesian optimisation and read coeffs from file",
  )

  args = parser.parse_args()
  return args


def plotDensityError(shape_lms, densityIn, densityMinus=0, tag="", axes=[0, 2]):
  """
  Plots the error in two gray-values (density) compared to each other.
  Could apply to density at LM location - density from model,
  or density from model - density from mean for example.
  Parameters
  ----------
  shape_lms (np.ndarray, NL, 3): shape for coordinates on scatter.
  densityIn (np.ndarray, NL): some density value for each landmark.
  densityMinus (np.ndarray, NL or float): a density value to subtract
  tag (str): some tag to include in file name
  """
  plt.close()
  fig, ax = plt.subplots(1, figsize=(16 / 2.54, 10 / 2.54))
  a = ax.scatter(
    shape_lms[:, axes[0]],
    shape_lms[:, axes[1]],
    cmap="seismic",
    c=abs(densityIn - densityMinus).reshape(-1),
    vmin=0,
    vmax=1,
    s=5,
  )
  ax.axes.xaxis.set_ticks([])
  ax.axes.yaxis.set_ticks([])
  # set colorbar
  cb = fig.colorbar(a, ax=ax)
  cb.set_label("density", fontsize=11)
  fig.suptitle("Density error in reconstruction", fontsize=12)
  fig.savefig(
    "./images/reconstruction/debug/density-error" + tag + ".png",
    pad_inches=0,
    format="png",
    dpi=300,
  )
  return None


def density_error_for_point_cloud(
  landmarks, imgCoords, img, comparison_density, tag="", axes=[0, 2]
):
  """
  For a given point cloud of landmarks, get density error on image and return
  vs modelled
  """
  density_at_lm = assam.getDensity(landmarks, img, imgCoords, axes=axes)
  plotDensityError(
    landmarks, density_at_lm, comparison_density, tag=tag, axes=axes
  )
  return None


def density_comparison(
  out_lms, gt_lms, img, imgCoords_out, density_at_lm_gt, tag="", axes=[0, 2]
):
  """
  For two given point clouds of landmarks, compare density and shape of each
  """
  density_at_lm_out = assam.getDensity(out_lms, img, imgCoords_out, axes=axes)
  # density_at_lm_gt = assam.getDensity(gt_lms, img, imgCoords_gt, axes=axes)
  plt.close()
  fig, ax = plt.subplots(1, 2, figsize=(16 / 2.54, 8 / 2.54))
  ax[0].scatter(
    out_lms[:, axes[0]],
    out_lms[:, axes[1]],
    c=density_at_lm_out,
    cmap="gray",
    s=2,
    vmin=-2,
    vmax=2,
  )
  ax[1].scatter(
    gt_lms[:, axes[0]],
    gt_lms[:, axes[1]],
    c=density_at_lm_gt,
    cmap="gray",
    s=2,
    vmin=-2,
    vmax=2,
  )
  ax[0].set_title("Output reconstruction")
  ax[1].set_title("Ground truth")
  print("out density", density_at_lm_out)
  print("gt density", density_at_lm_gt)
  for a in ax:
    a.axes.xaxis.set_ticks([])
    a.axes.yaxis.set_ticks([])
    a.axis("off")
  fig.savefig(
    "./images/reconstruction/debug/density-comparison" + tag + ".png",
    pad_inches=0,
    format="png",
    dpi=300,
  )
  return None


def getShapeParameters(
  average_landmarks, input_landmarks, shape_model, model_std
):
  """
  Output vector of shape parameters describing a shape with landmarks
  corresponding to those used to train the model.
  N_lms =  number of landmarks.
  N_train = number of samples in training set.

  Inputs:
  average_landmarks (np.ndarray N_lms,3): mean shape of training dataset for model
  input_landmarks (np.ndarray N_lms,3): shape to extract parameters from
  shape_model (np.ndarray N_train, 3N_lms): shape model extracted from
                                            PCA (eigenvectors)
  model_std (np.array N_train): vector of weightings for each principal component

  returns:
  shape_parameters (np.array N_train): vector with parameters describing the shape
  """
  input_landmarks_vec = input_landmarks.reshape(-1)
  input_landmarks_vec -= input_landmarks_vec.mean()
  input_landmarks_vec /= input_landmarks_vec.std()
  average_landmarks_vec = average_landmarks.reshape(-1)
  average_landmarks_vec -= average_landmarks_vec.mean()
  average_landmarks_vec /= average_landmarks_vec.std()

  shape_parameters = (
    np.dot(
      (input_landmarks_vec - average_landmarks_vec),
      shape_model[: len(model_std)].T,
    )
    / model_std
  )
  return shape_parameters


def newProjLMs(
  faces,
  meanNorms_face,
  surfCoords_mmOrig,
  surfCoords_mm,
  surfCoords_centred,
  landmarks,
  plane=1,
):
  """
  Get new landmarks that have a normal partially facing towards and away
  from the image. These are named 'projection' landmarks/'silhouette' landmarks

  Parameters
  ----------
  faces (np.ndarray/list?? N,3): point indexes making up each face on surface
  meanNorms_face : normal vectors for each point on face
  surfCoords_mmOrig : vertex coordinates in CT space
  surfCoords_mm : vertex coordinates in landmark space
  surfCoords_centred : vertex coordinates centred on the origin
  plane : projected plane to find silhouette LMs in

  """
  # IF MESH TOO FINE, THIS CAN BE PROBLEMATIC!
  projLM, projLM_ID = assam.getProjectionLandmarks(
    faces, meanNorms_face, surfCoords_mmOrig, plane
  )

  if plane == 1:
    projLM, projLM_ID = assam.deleteShadowedEdges(
      surfCoords_mm,
      projLM,
      projLM_ID,
    )

  if args.debug:
    key = "Airway"
    x = surfCoords_centred[key][projLM_ID[key]]
    plt.scatter(x[:, 0], x[:, 2], c="black", s=1)
    plt.scatter(
      landmarks["Airway"][:, 0],
      landmarks["Airway"][:, 2],
      c="blue",
      s=10,
      alpha=0.3,
    )
    plt.show()
  # reorder projected surface points to same order as landmarks
  # print("number proj airway pts", len(assam.projLM_ID["Airway"]))
  for key in shapes:
    if not args.quiet:
      print("reordering projected landmarks for", key)
    newProjIDs = []
    newProjPos = []
    # find closest surface point for each landmark
    for p, point in enumerate(surfCoords_centred[key][projLM_ID[key]]):
      dist = utils.euclideanDist(landmarks[key], point)
      closest_lm_index = np.argmin(dist)
      # prevent duplicating points
      if closest_lm_index not in newProjIDs:
        newProjIDs.append(closest_lm_index)
        newProjPos.append(landmarks[key][closest_lm_index, [0, 2]])
    projLM_ID[key] = copy(newProjIDs)
    projLM[key] = copy(np.vstack(newProjPos))
  return projLM, projLM_ID


def getMeanGraph(
  caseIDs,
  landmarks,
  mean_landmarks,
  graph_files="landmarks/manual-jw-diameterFromSurface/nxGraph{}landmarks.pickle",
):
  """
  Output networkx graph with mean position and additional metadata on
  how each graph node matches landmark location in numpy array

  Inputs:
  caseIDs (list, str)  ordered list of IDs for each training case.
  landmarks (np.ndarray, N_train, N_lms, 3): all landmarks for all patients
  mean_landmarks (np.ndarray, N_lms, 3): landmarks averaged over all patients
  graph_files (str): string used to search for landmarked graphs by caseID

  returns:
  lgraphMean (nx.DiGraph): directed graph with mean landmark position at each node
  """
  lgraphList = [nx.read_gpickle(graph_files.format(cID)) for cID in caseIDs]
  posList = []
  for i, lgraph in enumerate(lgraphList):
    pos = []
    for node in lgraph.nodes:
      pos.append(lgraph.nodes[node]["pos"])
    pos = np.vstack(pos)
    posList.append(pos)
  lgraphMean = lgraphList[-1].copy()
  lgraph = lgraphList[-1].copy()
  for node in lgraph.nodes:
    pos = lgraph.nodes[node]["pos"]
    dist = utils.euclideanDist(landmarks[i], pos)
    currentLM = np.argmin(dist)
    # find closest graph node, if it is not a landmark then find next closest
    isin = np.isclose(posList[i], landmarks[i][currentLM]).all(axis=1)
    while isin.sum() == 0:
      dist[currentLM] = 100000000
      currentLM = np.argmin(dist)
      isin = np.isclose(posList[i], landmarks[i][currentLM]).all(axis=1)
    # assign metadata to graphs
    lgraph.nodes[node]["npID"] = currentLM
    lgraphMean.nodes[node]["pos"] = mean_landmarks[currentLM]
    lgraphMean.nodes[node]["npID"] = currentLM
  return lgraphMean


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


if __name__ == "__main__":
  args = getInputs()
  date_today = str(date.today())
  if not args.quiet:
    print(__doc__)
  startTime = time()

  landmarkDir = args.inp
  case = args.case
  tag = args.out
  describedVariance = args.var
  drrDir = args.drrs
  debug = args.debug
  shapes = args.shapes.split()
  surfDir = args.meshdir
  numEpochs = args.epochs
  xrayEdgeFile = args.xray
  if args.bayesian_opt:
    bayes_pts = np.loadtxt("points_to_sample.txt")
    c_dense = bayes_pts[0]
    c_edge = bayes_pts[1]
    c_prior = c_prior = (1.0e-4) ** (bayes_pts[2])
  else:
    c_dense = args.c_dense
    c_edge = args.c_edge
    c_prior = args.c_prior
  print(f"c_dense is {c_dense}")
  print(f"c_edge is {c_edge}")
  print(f"c_prior is {c_prior}")
  c_anatomical = args.c_anatomical
  c_grad = args.c_grad
  kernel_radius = args.kernel_radius
  kernel_distance = args.kernel_distance
  imgSpaceCoeff = args.imgSpacing

  shapes = ["Airway", "RUL", "RML", "RLL", "LUL", "LLL"]
  # shapes = ['Airway']
  lobes = ["RUL", "RML", "RLL", "LUL", "LLL"]

  # clean up shapes not given as input args
  delInd = []
  for i, lobe in enumerate(lobes):
    if lobe not in args.shapes.split():
      delInd.append(i)
  for ind in delInd[::-1]:
    lobes.pop(ind)
  delInd = []
  for i, shape in enumerate(shapes):
    if shape not in args.shapes.split():
      delInd.append(i)
  for ind in delInd[::-1]:
    shapes.pop(ind)
  del delInd

  # numbering for each lobe in file
  lNums = {"RUL": "4", "RML": "5", "RLL": "6", "LUL": "7", "LLL": "8"}

  # landmarkDir, case, tag, describedVariance, drrDir, debug, \
  #         shapes, surfDir, numEpochs, xrayEdgeFile, \
  #         c_edge, c_dense, c_prior, imgSpaceCoeff = getInputs()
  img = None
  spacing_xr = None

  print("\tReading data")
  with open("config_3proj.json") as f:
    config = hjson.load(f)
  # read DRR data
  originDirs = filesFromRegex(config["luna16paths"]["origins"])
  spacingDirs = filesFromRegex(config["luna16paths"]["spacing"])
  imDirs = filesFromRegex(config["luna16paths"]["drrs"])
  imDirs_left = filesFromRegex(config["luna16paths"]["drrs_left"])
  imDirs_right = filesFromRegex(config["luna16paths"]["drrs_right"])
  imDirs_45 = filesFromRegex(config["luna16paths"]["drrs_-45"])
  patientIDs = matchesFromRegex(config["luna16paths"]["origins"])

  # read landmark data
  landmarkDirs = filesFromRegex(config["luna16paths"]["landmarks"])
  lmIDs = matchesFromRegex(config["luna16paths"]["landmarks"])
  landmarkDirsOrig = glob("landmarks/manual-jw/landmarks*.csv")
  landmarkDirs.sort()
  landmarkDirsOrig.sort()

  assert (
    len(spacingDirs)
    == len(imDirs)
    == len(originDirs)
    == len(imDirs_left)
    == len(imDirs_right)
    == len(imDirs_45)
  ), (
    "Error reading image data. "
    f"Num spacing dirs = {len(spacingDirs)}. "
    f"Num imDirs dirs = {len(imDirs)}. "
    f"Num originDirs dirs = {len(originDirs)}. "
    f"Num imDirs_left dirs = {len(imDirs_left)}. "
    f"Num imDirs_right dirs = {len(imDirs_right)}. "
    f"Num imDirs_45 dirs = {len(imDirs_45)}. "
  )

  if (
    len(imDirs) == 0
    or len(originDirs) == 0
    or len(landmarkDirs) == 0
    or len(spacingDirs) == 0
    or len(landmarkDirsOrig) == 0
  ):
    print(
      "ERROR: The directories you have declared are empty.",
      "\nPlease check your input arguments.",
    )
    exit()
  # transDirs = glob( "savedPointClouds/allLandmarks/"
  #                             +"transformParams_case*"
  #                             +"_m_"+shape+".dat")
  # transDirs_all = glob( "savedPointClouds/allLandmarks/"
  #                             +"transformParams_case*"
  #                             +"_m_"+shape+".dat")

  # remove scans without landmarks from DRR dirs
  missing = []
  missingID = []
  delInd = []
  for i, currID in enumerate(patientIDs):
    if currID not in lmIDs:
      delInd.append(i)

  for dId in delInd[::-1]:
    originDirs.pop(dId)
    spacingDirs.pop(dId)
    imDirs.pop(dId)
    imDirs_left.pop(dId)
    imDirs_right.pop(dId)
    imDirs_45.pop(dId)
    patientIDs.pop(dId)
  missing = []
  missingID = []
  # exit()
  # for p, pID in enumerate(patientIDs):
  #   if pID not in imDirs:
  #     missing.append(p)
  #     missingID.append(pID)
  # '''loop in reverse so no errors due to index being deleted
  #     i.e. delete index 12, then next item to delete is 21,
  #     but beca  use a previous element has been removed, you delete item 20'''
  # for m in missing[::-1]:
  #   landmarkDirs.pop(m)
  #   patientIDs.pop(m)
  #   originDirs.pop(m)
  #   spacingDirs.pop(m)
  #   imDirs.pop(m)

  landmarks = np.array(
    [np.loadtxt(l, delimiter=",", skiprows=1) for l in landmarkDirs]
  )
  nodalCoordsOrig = np.array(
    [
      np.loadtxt(l, delimiter=",", skiprows=1, usecols=[1, 2, 3])
      for l in landmarkDirsOrig
    ]
  )

  lmOrder = dict.fromkeys(shapes)
  lmOrder["SKELETON"] = np.loadtxt(
    landmarkDir + "landmarkIndexSkeleton.txt"
  ).astype(int)
  lmOrder["LUNGS"] = []
  for shape in shapes:
    lmOrder[shape] = np.loadtxt(
      landmarkDir + f"landmarkIndex{shape}.txt", dtype=int
    )
    if shape in lobes:
      lmOrder["LUNGS"].extend(list(lmOrder[shape]))
  lmOrder["LUNGS"] = np.array(lmOrder["LUNGS"])

  lgraph = getMeanGraph(patientIDs, landmarks, landmarks.mean(axis=0))
  nx.write_gpickle(lgraph, "skelGraphs/nxGraphLandmarkMean.pickle")
  lgraph_branches = utils.simplifyGraph(lgraph)
  nx.write_gpickle(
    lgraph_branches, "skelGraphs/nxGraphLandmarkMeanBranchesOnly.pickle"
  )
  # extra code below to visualise graph if desired
  # lines = []
  # vp = v.Plotter()
  # for edge in lgraph.edges:
  #   l = v.Line(lgraph.nodes[edge[0]]['pos'], lgraph.nodes[edge[1]]['pos'],
  #               lw=4, c='black')
  #   lines.append(l)
  # pNodes = v.Points(landmarks.mean(axis=0)[lmOrder['SKELETON']],r=10,c='black')
  # vp.show(lines,pNodes)
  # vp.show(interactive=True)

  # read appearance modelling data
  origin = np.vstack([np.loadtxt(o, skiprows=1)] for o in originDirs)
  spacing = np.vstack(
    [np.loadtxt(o, skiprows=1) * imgSpaceCoeff] for o in spacingDirs
  )
  # crop last two rows of pixels off XR so white pixels don't interfere with normalising
  drrArr = np.rollaxis(
    # np.dstack([utils.loadXR(o)[:-2,:-2][::imgSpaceCoeff,::imgSpaceCoeff]
    np.dstack(
      [utils.loadXR(o)[::imgSpaceCoeff, ::imgSpaceCoeff] for o in imDirs]
    ),
    2,
    0,
  )
  if "drrs_left" in config["training"]["img_keys"]:
    drrArr_left = np.rollaxis(
      # np.dstack([utils.loadXR(o)[:-2,:-2][::imgSpaceCoeff,::imgSpaceCoeff]
      np.dstack(
        [utils.loadXR(o)[::imgSpaceCoeff, ::imgSpaceCoeff] for o in imDirs_left]
      ),
      2,
      0,
    )
  if "drrs_right" in config["training"]["img_keys"]:
    drrArr_right = np.rollaxis(
      # np.dstack([utils.loadXR(o)[:-2,:-2][::imgSpaceCoeff,::imgSpaceCoeff]
      np.dstack(
        [
          utils.loadXR(o)[::imgSpaceCoeff, ::imgSpaceCoeff]
          for o in imDirs_right
        ]
      ),
      2,
      0,
    )
  if "drrs_-45" in config["training"]["img_keys"]:
    drrArr_45 = np.rollaxis(
      # np.dstack([utils.loadXR(o)[:-2,:-2][::imgSpaceCoeff,::imgSpaceCoeff]
      np.dstack(
        [utils.loadXR(o)[::imgSpaceCoeff, ::imgSpaceCoeff] for o in imDirs_45]
      ),
      2,
      0,
    )

  if config["training"]["num_imgs"] == 2:
    # join so array is shape Npatients, Nimages, Npixels_x, Npixels_y
    drrArr = np.stack((drrArr, drrArr_left), axis=1)
  elif config["training"]["num_imgs"] == 3:
    drrArr = np.stack((drrArr, drrArr_left, drrArr_right), axis=1)

  # offset centered coordinates to same reference frame as CT data
  carinaArr = nodalCoordsOrig[:, 1]
  landmarks_in_ct_space = landmarks + carinaArr[:, np.newaxis]

  # load pre-prepared mean stl and point cloud of all data
  meanArr = landmarks.mean(axis=0)

  # reset new params
  landmarksAll = 0
  origin = copy(origin)
  spacing = copy(spacing)
  drrArr = copy(drrArr)
  # landmarks = landmarks_in_ct_space.copy()

  # format data for testing by randomising selection and removing these
  # from training
  assignedTestIDs = ["0645", "3948", "5268", "6730", "8865"]
  assignedTestIDs = ["3948"]
  assignedTestIDs = [case]
  testSize = args.testSize
  testID = []
  randomise_testing = args.randomise  # False
  default_patientIDs = copy(patientIDs)
  if randomise_testing or assignedTestIDs[0] == "none":
    assignedTestIDs = []
    testSet = np.random.randint(0, len(patientIDs) - 1, testSize)
    # check if test and training datasets share overlapping samples
    testOverlapTrain = True
    while np.unique(testSet).size != testSize and testOverlapTrain:
      testSet = np.random.randint(0, len(patientIDs) - 1, testSize)
      testOverlapTrain = [True for p in testID if p in assignedTestIDs]

      for t in testSet[::-1]:
        # store test data in different list
        testID.append(patientIDs[t])
    print("randomising testing")
  else:
    testSet = []
    # assignedTestIDs = ["9484"]
    # assignedTestIDs = ["5268"]
    for i, pID in enumerate(patientIDs):
      if pID in assignedTestIDs:
        testSet.append(i)

  testLM = []
  testOrigin = []
  testSpacing = []
  testIm = []
  testSet.sort()
  landmarks_in_ct_spaceDef = landmarks_in_ct_space.copy()
  landmarksDef = landmarks.copy()
  ground_truth_landmarks = []
  ground_truth_landmarks_in_ct_space = []
  for t in testSet[::-1]:
    # store test data in different list
    testID.append(patientIDs[t])
    testOrigin.append(origin[t])
    testSpacing.append(spacing[t])
    testIm.append(drrArr[t])
    testLM.append(copy(landmarks[t]))
    ground_truth_landmarks.append(copy(landmarks[t]))
    # for post-processing
    ground_truth_landmarks_in_ct_space.append(copy(landmarks_in_ct_space[t]))
    """ """
    if args.keep_in_training:
      pass
    else:
      # remove test data from train data
      patientIDs.pop(t)
      origin = np.delete(origin, t, axis=0)
      spacing = np.delete(spacing, t, axis=0)
      drrArr = np.delete(drrArr, t, axis=0)
      landmarks = np.delete(landmarks, t, axis=0)
      landmarks_in_ct_space = np.delete(landmarks_in_ct_space, t, axis=0)
    """ """

  if randomise_testing:
    print("test IDs are", testID)
  assert len(testID) != 0, "no testID selected"

  meanArr = landmarks.mean(axis=0)

  # template_lm_file_lobes = landmarkDir+"/landmarks0{}-case8684.csv"

  # create appearance model instance and load data
  ssam = RespiratorySSAM(
    landmarks,
    landmarks_in_ct_space,
    drrArr,
    origin,
    spacing,
    train_size=landmarks.shape[0],
    rotation=config["training"]["rotation"],
    img_coords_axes=config["training"]["img_axes"],
  )
  if testIm[0].ndim == 3:
    # if multiple images for reconstruction, get density for each one
    # which is last N columns from xg_train
    density = ssam.xg_train[:, :, -testIm[0].shape[0] :]
    number_of_features = 3 + testIm[0].shape[0]  # 3 = num coordinates
  else:
    density = ssam.xg_train[:, :, -1].reshape(len(landmarks), -1, 1)
    number_of_features = 4  # three coordinates, and a density value

  model = ssam.phi_sg
  meanArr = np.mean(landmarks, axis=0)

  # set number of modes
  numModes = np.where(
    np.cumsum(ssam.pca_sg.explained_variance_ratio_) > describedVariance
  )[0][0]
  if not args.quiet:
    print("modes used is", numModes)

  # center the lobes vertically
  # keep vertical alignment term for later use
  lmAlign = meanArr[:, 2].mean()  # landmarks[:,2].mean()
  offset_to_centre = meanArr.mean(axis=0)
  meanArr[:, 2] -= lmAlign

  modelDict = dict.fromkeys(shapes)
  inputCoords = dict.fromkeys(shapes)
  inputCoords["ALL"] = meanArr.copy()
  modelDict["ALL"] = model
  for shape in shapes:
    inputCoords[shape] = copy(meanArr[lmOrder[shape]])
    modelDict[shape] = model.reshape(len(landmarks), -1, number_of_features)[
      :, lmOrder[shape]
    ]

  mean_mesh = dict.fromkeys(shapes)
  faces = dict.fromkeys(shapes)  # faces of mean surface for each shape
  surfCoords_centred = dict.fromkeys(shapes)
  surfCoords_mmOrig = dict.fromkeys(shapes)  # surface nodes for each shape
  surfCoords_mm = dict.fromkeys(shapes)  # surface nodes in same ordering as LMs
  meanNorms_face = dict.fromkeys(
    shapes
  )  # normals for each face (?) of mean mesh
  meanNorms_face_rot45 = dict.fromkeys(shapes)
  # surfCoords_mmOrig_rot45 = dict.fromkeys(shapes)
  # surfCoords_mm_rot45 = dict.fromkeys(shapes)
  surfToLMorder = dict.fromkeys(shapes)  # mapping between surface nodes and LMs
  newMean = args.newMean
  # create mesh for population average from a morphing algorithm
  templateDir = "templates/coarserTemplates/"
  mean_shape_file = templateDir + "meanAirway.stl"

  # assert  glob(mean_shape_file) != 0 or not newMean, 'error in loading meshes. We created coarsened ones manually - do not overwrite'
  if glob(mean_shape_file) == 0 or newMean:
    print("making airway template mesh")
    template_lmFileOrig = "landmarks/manual-jw/landmarks3948.csv"
    # template_lmFile = 'landmarks/manual-jw-diameterFromSurface/landmarks3948_diameterFromSurf.csv'
    template_lmFile = "allLandmarks/allLandmarks3948.csv"
    template_airway_file = (
      "segmentations/template3948/smooth_newtemplate3948_mm.stl"
    )
    lm_template = np.loadtxt(
      template_lmFile, skiprows=1, delimiter=",", usecols=[0, 1, 2]
    )
    carinaTemplate = (
      np.loadtxt(
        template_lmFileOrig, skiprows=1, delimiter=",", usecols=[1, 2, 3]
      )[1]
      * -1
    )
    template_airway_mesh = v.load(template_airway_file)
    template_airway_mesh = template_airway_mesh.pos(carinaTemplate)
    morph_airway = MorphAirwayTemplateMesh(
      lm_template[lmOrder["Airway"]],
      meanArr[lmOrder["Airway"]],
      template_airway_mesh,
      sigma=0.3,
      quiet=args.quiet,
    )
    morph_airway.mesh_target.write(mean_shape_file)
    np.savetxt(
      templateDir + "meanAirway.csv",
      meanArr[lmOrder["Airway"]],
      delimiter=",",
      header="x, y, z",
    )
    for lobe in lobes:
      print(f"making {lobe} template mesh")
      template_lmFile = "allLandmarks/allLandmarks8684.csv"
      lm_template = np.loadtxt(
        template_lmFile, skiprows=1, delimiter=",", usecols=[0, 1, 2]
      )
      template_mesh_file_lobes = "/home/josh/3DSlicer/project/luna16Rescaled/case8684_smooth/8684_mm_{}.stl"
      # template_mesh_file_lobes = "/home/josh/3DSlicer/project/luna16Rescaled/case3948/3948_mm_{}.stl"
      # lm_template_lobes = np.loadtxt(template_lm_file_lobes.format(key), delimiter=",")
      template_lobe_mesh = v.load(template_mesh_file_lobes.format(lNums[lobe]))
      morph_lobe = MorphLobarTemplateMesh(
        lm_template[lmOrder[lobe]],
        meanArr[lmOrder[lobe]],
        template_lobe_mesh,
        sigma=0.3,
        quiet=args.quiet,
      )
      mean_lobe_file_out = templateDir + f"mean{lobe}.stl"
      morph_lobe.mesh_target.write(mean_lobe_file_out)
      np.savetxt(
        templateDir + f"mean{lobe}.csv",
        meanArr[lmOrder[lobe]],
        delimiter=",",
        header="x, y, z",
      )

  for key in shapes:
    mean_shape_file = templateDir + f"mean{key}.stl"
    assert (
      len(glob(mean_shape_file)) > 0
    ), f"file {mean_shape_file} does not exist!"

    mean_mesh[key] = v.load(mean_shape_file).computeNormals()
    # mesh_template_lms[key] =

  # reorder unstructured stl file to be coherent w/ model and landmarks
  # extract mesh data (coords, normals and faces)
  for key in shapes:
    if not args.quiet:
      print(f"loading {key} mesh")
    if not args.quiet:
      print("original num cells", len(mean_mesh[key].faces()))
    if key == "Airway":
      mesh = mean_mesh[key].clone()
      pass
      # mesh = mean_mesh[key].clone().decimate(N=40e3).clean()
    else:
      mesh = mean_mesh[key].clone().decimate(fraction=0.1).clean()
    if not args.quiet:
      print("decimated num cells", len(mesh.faces()))
    mesh_45 = mesh.clone().rotateZ(45)
    meanNorms_face_rot45[key] = mesh_45.normals(cells=True)
    # vp = v.Plotter()
    # vp += mesh_45.alpha(0.8)
    # vp += mesh.alpha(0.2)
    # vp.show()
    # exit()
    # load mesh data and create silhouette
    surfCoords = mesh.points()
    meanNorms_face[key] = mesh.normals(cells=True)
    faces[key] = np.array(mesh.faces())

    # offset to ensure shapes are aligned to carina
    surfCoords_centred[key] = copy(surfCoords)
    surfCoords_mm[key] = surfCoords + carinaArr.mean(axis=0)
    surfCoords_mmOrig[key] = copy(surfCoords_mm[key])
    # surfCoords_mm_rot45[key]
    surfToLMorder[key] = []
    for point in meanArr[lmOrder[key]]:
      surfToLMorder[key].append(
        np.argmin(utils.euclideanDist(surfCoords, point))
      )
    surfCoords_mm[key] = surfCoords_mm[key][surfToLMorder[key]]

  tagBase = copy(tag)
  save_initial_coords = copy(inputCoords)
  for t, (
    target_id,
    target_lm,
    target_img,
    target_origin,
    target_spacing,
  ) in enumerate(zip(testID, testLM, testIm, testOrigin, testSpacing)):

    tag = tagBase + "_case" + target_id
    # load image directly from training data
    img = target_img.copy()

    # index 0 as output is stacked
    imgCoords = ssam.sam.drrArrToRealWorld(img, np.zeros(3), target_spacing)[0]
    spacing_xr = target_spacing.copy()
    # center image coords, so in the same coord system as edges
    imgCoords -= np.mean(imgCoords, axis=0)

    # edge points in units of pixels from edge map
    edgePoints = [None] * len(config["test-set"]["outlines"])
    for f, file in enumerate(config["test-set"]["outlines"]):
      file_re = config["test-set"]["outlines"][f].format(target_id, target_id)
      print(file_re)
      edgePoints[f] = np.loadtxt(file_re, delimiter=",")
      edgePoints[f] = np.unique(edgePoints[f], axis=0)
    # if only 1 x-ray given, change shape from list of 2D arrays to one 2D array
    if len(edgePoints) == 1:
      edgePoints = edgePoints[f]
    edgePoints_contrast = np.loadtxt(
      config["test-set"]["contrast-outline"][0].format(target_id, target_id),
      delimiter=",",
    )

    # debug checks to show initial alignment of image with edge map,
    # and mean shape with edge map
    if debug:
      if config["training"]["num_imgs"] == 1:
        # for plotting image in same ref frame as the edges
        axes = config["training"]["img_axes"][0]
        extent = [
          imgCoords[:, axes[0]].min(),
          imgCoords[:, axes[0]].max(),
          imgCoords[:, axes[1]].min(),
          imgCoords[:, axes[1]].max(),
        ]

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img, cmap="gray", extent=extent)
        ax[0].scatter(edgePoints[:, 0], edgePoints[:, 1], s=2)
        ax[1].scatter(meanArr[:, 0], meanArr[:, 2], s=1, c="black")
        ax[1].scatter(edgePoints[:, 0], edgePoints[:, 1], s=2, c="blue")
        plt.show()
      else:
        for i in range(0, config["training"]["num_imgs"]):
          img_ax_i = config["training"]["img_axes"][i]
          extent = [
            imgCoords[:, img_ax_i[0]].min(),
            imgCoords[:, img_ax_i[0]].max(),
            imgCoords[:, img_ax_i[1]].min(),
            imgCoords[:, img_ax_i[1]].max(),
          ]
          plt.close()
          fig, ax = plt.subplots(1, 2)
          ax[0].imshow(img[i], cmap="gray", extent=extent)
          if config["training"]["rotation"][i] == 0:
            ax[0].scatter(edgePoints[i][:, 0], edgePoints[i][:, 1], s=2)
            ax[1].scatter(edgePoints[i][:, 0], edgePoints[i][:, 1], s=2)
            ax[1].scatter(
              meanArr[:, config["training"]["img_axes"][i][0]],
              meanArr[:, config["training"]["img_axes"][i][1]],
              s=1,
              c="black",
            )
          else:
            coords_rot = utils.rotate_coords_about_z(
              meanArr, config["training"]["rotation"][i]
            )

            ax[1].scatter(edgePoints[i][:, 0], edgePoints[i][:, 1], s=2)
            ax[0].scatter(
              coords_rot[:, config["training"]["img_axes"][i][0]],
              coords_rot[:, config["training"]["img_axes"][i][1]],
              s=1,
              c="yellow",
              alpha=0.2,
            )
            ax[1].scatter(edgePoints[i][:, 0], edgePoints[i][:, 1], s=2)
          plt.show()
      # check we get correct gray value for the rotated x-rays
      for i in range(0, config["training"]["num_imgs"]):
        img_centre = target_origin + (target_spacing * 500.0) / 2
        coords_rot = utils.rotate_coords_about_z(
          copy(ground_truth_landmarks_in_ct_space[t]),
          config["training"]["rotation"][i],
          img_centre,
        )
        img_coords_ground_truth = ssam.sam.drrArrToRealWorld(
          target_img.reshape(-1, 500, 500)[i], target_origin, target_spacing
        )[0]

        ground_truth_density = ssam.sam.getDensity(
          coords_rot,
          target_img.reshape(-1, 500, 500)[i],
          img_coords_ground_truth,
          config["training"]["img_axes"][i],
        )

        plt.scatter(
          coords_rot[:, config["training"]["img_axes"][i][0]],
          coords_rot[:, 2],
          c=ground_truth_density,
          cmap="gray",
        )
        plt.show()

    # declare posterior shape model class
    if strtobool(config["test-set"]["contrast"][0]) == True:
      assam = RespiratoryReconstructSSAM(
        shape=inputCoords,
        xRay=edgePoints,
        lmOrder=lmOrder,
        normals=None,
        transform=carinaArr[t] * 0.0,
        img=copy(img),
        # imgSpacing=spacing_xr,
        imgCoords=imgCoords,
        imgCoords_all=ssam.sam.imgCoords_all,
        imgCoords_axes=config["training"]["img_axes"],
        density=density,
        model=modelDict,
        modeNum=numModes,
        c_edge=c_edge,
        c_prior=c_prior,
        c_dense=c_dense,
        c_grad=c_grad,
        c_anatomical=c_anatomical,
        kernel_distance=kernel_distance,
        kernel_radius=kernel_radius,
        quiet=args.quiet,
        img_names=config["training"]["img_names"],
        shapes_to_skip_fitting=config["training"]["shapes_to_skip_fit"],
        plot_freq=args.plot_freq,
        plot_tag=f"case{target_id}",
        rotation=config["training"]["rotation"],
        contrast_outline=edgePoints_contrast,
      )
    else:
      assam = RespiratoryReconstructSSAM(
        shape=inputCoords,
        xRay=edgePoints,
        lmOrder=lmOrder,
        normals=None,
        transform=carinaArr[t] * 0.0,
        img=copy(img),
        # imgSpacing=spacing_xr,
        imgCoords=imgCoords,
        imgCoords_all=ssam.sam.imgCoords_all,
        imgCoords_axes=config["training"]["img_axes"],
        density=density,
        model=modelDict,
        modeNum=numModes,
        c_edge=c_edge,
        c_prior=c_prior,
        c_dense=c_dense,
        c_grad=c_grad,
        c_anatomical=c_anatomical,
        kernel_distance=kernel_distance,
        kernel_radius=kernel_radius,
        quiet=args.quiet,
        img_names=config["training"]["img_names"],
        shapes_to_skip_fitting=config["training"]["shapes_to_skip_fit"],
        plot_freq=args.plot_freq,
        plot_tag=f"case{target_id}",
        rotation=config["training"]["rotation"],
      )
    assam.spacing_xr = spacing_xr
    # import variables to class
    assam.variance = ssam.variance[:numModes]
    assam.std = ssam.std[:numModes]
    # import functions to PSM class
    assam.getg_allModes = ssam.sam.getg_allModes
    assam.getDensity = ssam.sam.getDensity
    assam.normaliseTestImageDensity = ssam.sam.normaliseTestImageDensity

    print("getting projected landmarks")
    projLM_file = "allLandmarks/projectedMeanLandmarks{}.csv"
    projLM_ID_file = "allLandmarks/projectedMeanLandmarksID{}.csv"
    t1 = time()

    # True if given new mesh to get projection landmarks.
    if len(glob(projLM_file.format("*"))) == 0 or args.newProjLM:
      projLM, projLM_ID = newProjLMs(
        faces,
        meanNorms_face,
        surfCoords_mmOrig,
        surfCoords_mm,
        surfCoords_centred,
        inputCoords,
      )
      # write new projection coords to text file
      for key in projLM.keys():
        np.savetxt(
          projLM_file.format(key),
          projLM[key],
          header="x y",
          delimiter=",",
        )
        np.savetxt(
          projLM_ID_file.format(key),
          projLM_ID[key],
          header="ID",
          fmt="%i",
        )
      assam.projLM, assam.projLM_ID = projLM.copy(), projLM_ID.copy()

      if config["training"]["num_imgs"] > 1:
        projLM_ID_multipleproj = [None] * config["training"]["num_imgs"]
        projLM_ID_multipleproj[0] = copy(projLM_ID)
        for proj_i in range(1, config["training"]["num_imgs"]):
          if config["training"]["rotation"][proj_i] == 0:
            # find which plane is projected
            plane = np.setdiff1d(
              [0, 1, 2], config["training"]["img_axes"][proj_i]
            )[0]
            _, projLM_ID_multipleproj[proj_i] = newProjLMs(
              faces,
              meanNorms_face,
              surfCoords_mmOrig,
              surfCoords_mm,
              surfCoords_centred,
              inputCoords,
              plane=0,
            )
          elif config["training"]["rotation"][proj_i] == 45:
            _, projLM_ID_multipleproj[proj_i] = newProjLMs(
              faces,
              meanNorms_face_rot45,
              surfCoords_mmOrig,
              surfCoords_mm,
              surfCoords_centred,
              inputCoords,
              plane=0,
            )
          # get projection IDs for alternative images (i.e. additional lateral views)
        assam.projLM_ID_multipleproj = copy(projLM_ID_multipleproj)
        # save projLM IDs for loading on future runs
        for proj_ind, proj in enumerate(assam.projLM_ID_multipleproj):
          for key in proj.keys():
            np.savetxt(
              config["luna16paths"]["projLM_ID_file"].format(key, proj_ind),
              proj[key],
              header="ID",
              fmt="%i",
            )
            print(config["luna16paths"]["projLM_ID_file"].format(key, proj_ind))
            print(proj[key][:10])
    else:
      # if no new projLMs are needed, load some from previously created csv files
      assam.projLM = dict.fromkeys(shapes)
      assam.projLM_ID = dict.fromkeys(shapes)
      # if multiple projections are used, create a separate dict to keep
      # code simple when switching between 1 or 2 projections
      assam.projLM_ID_multipleproj = [
        dict.fromkeys(shapes) for i in range(0, config["training"]["num_imgs"])
      ]
      # load files for each shape used
      for key in shapes:
        assam.projLM[key] = np.loadtxt(
          projLM_file.format(key), skiprows=1, delimiter=","
        )
        assam.projLM_ID[key] = np.loadtxt(
          projLM_ID_file.format(key), dtype=int, skiprows=1
        )
        if config["training"]["num_imgs"] == 1:
          continue
        # load projected landmarks for all projections
        for proj_ind in range(0, config["training"]["num_imgs"]):
          print(config["luna16paths"]["projLM_ID_file"].format(key, proj_ind))
          assam.projLM_ID_multipleproj[proj_ind][key] = np.loadtxt(
            config["luna16paths"]["projLM_ID_file"].format(key, proj_ind),
            dtype=int,
            skiprows=1,
          )
        if debug:
          # check projLM_ID_multipleproj are different and not overwritten
          for proj_ind in range(0, config["training"]["num_imgs"]):
            print(
              "after loop",
              proj_ind,
              assam.projLM_ID_multipleproj[proj_ind][key][:10],
            )

    print(f"time taken to get projected points {round(time() - t1)} s")
    print(f"finished getting projected landmarks. Time taken = {time() - t1} s")

    assam.fissureLM_ID = 0
    t1 = time()

    assam.projLM_IDAll = []
    pointCounter = 0
    for key in assam.projLM_ID.keys():
      print(f"\n{key}")
      if key not in ["Airway", "RML"]:
        assam.projLM_IDAll.extend(
          list(np.array(assam.projLM_ID[key]) + pointCounter)
        )
        # tmp_id = np.arange(0, inputCoords[key].shape[0], 1)
        # assam.projLM_IDAll.extend(tmp_id+pointCounter)
      pointCounter += inputCoords[key].shape[0]
    assam.projLM_IDAll = np.array(assam.projLM_IDAll)

    # debug check showing projected landmarks (i.e. the outline)
    # this should be a smooth line in the curvature of the lobes
    if debug:
      plot_pts = surfCoords_mm["Airway"][assam.projLM_ID["Airway"]]
      print("number proj airway pts", len(assam.projLM_ID["Airway"]))
      plt.close()
      plt.scatter(plot_pts[:, 0], plot_pts[:, 2])
      for key in lobes:
        plt.scatter(assam.projLM[key][:, 0], assam.projLM[key][:, 1])
        # plt.scatter(projlm_basenew[key][:,0], projlm_basenew[key][:,1])
        # plt.scatter(projlm_base[key][:,0], projlm_base[key][:,1])
      plt.show()
      plt.close()
      # plt.scatter(plot_pts[:,0], plot_pts[:,2])
      if config["training"]["num_imgs"] > 1:
        for proj_ind in range(0, config["training"]["num_imgs"]):
          print(config["training"]["img_axes"][proj_ind])
          for key in lobes:
            plt.scatter(
              inputCoords[key][
                assam.projLM_ID_multipleproj[proj_ind][key],
                config["training"]["img_axes"][proj_ind][0],
              ],
              inputCoords[key][
                assam.projLM_ID_multipleproj[proj_ind][key],
                config["training"]["img_axes"][proj_ind][1],
              ],
            )
          plt.show()
    # initialise parameters to be optimised - including initial values + bounds
    optTrans_new = dict.fromkeys(["pose", "scale"])
    optTrans_new["pose"] = [0, 0]
    optTrans_new["scale"] = 1
    initPose = np.array([optTrans_new["pose"][0], optTrans_new["pose"][1], 1])
    bounds = np.array(
      [
        (-20, 20),
        (-20, 20),
        # (0.7, 1.3),
        (0.4, 2.0),  # (optTrans_new["scale"]*0.9, optTrans_new["scale"]*1.1),
        (-3, 3),
      ]
    )

    # initialise parameters that control optimisation process
    assam.optimiseStage = "both"  # tell class to optimise shape and pose
    assam.optIterSuc, assam.optIter = 0, 0
    assam.scale = optTrans_new["scale"]
    assert len(assam.projLM_ID) != 0, "no projected landmarks"

    # for debugging input data, run only up to just before optimisation starts
    if args.load_data_only:
      exit()

    # perform optimisation
    optAll = assam.optimiseAirwayPoseAndShape(
      assam.objFuncAirway, initPose, bounds, epochs=numEpochs, threads=1
    )
    print(
      "\n\n\n\n\n\t\t\tTime taken is {0} ({1} mins)".format(
        time() - startTime, round((time() - startTime) / 60.0), 3
      )
    )

    # get final shape
    outShape = assam.morphAirway(
      inputCoords["ALL"],
      inputCoords["ALL"].mean(axis=0),
      optAll["b"],
      assam.model_s["ALL"][: len(optAll["b"])],
    )
    outShape = assam.centerThenScale(
      outShape, optAll["scale"], outShape.mean(axis=0)
    )

    out_file = "{}_{}.{}"
    out_surf_file = "surfaces/" + out_file
    out_lm_file = "outputLandmarks/" + out_file
    np.savetxt(
      out_lm_file.format(tag, "ALL", "csv"),
      outShape,
      header="x, y, z",
      delimiter=",",
    )

    for i, (axes, rotation_angle) in enumerate(
      zip(config["training"]["img_axes"], config["training"]["rotation"])
    ):
      extent = [
        imgCoords[:, axes[0]].min(),
        imgCoords[:, axes[0]].max(),
        imgCoords[:, axes[1]].min(),
        imgCoords[:, axes[1]].max(),
      ]
      coords_rot = utils.rotate_coords_about_z(copy(outShape), rotation_angle)
      plt.close()
      plt.imshow(img.reshape(-1, 500, 500)[i], cmap="gray", extent=extent)
      plt.scatter(
        coords_rot[:, axes[0]],
        coords_rot[:, axes[1]],
        s=2,
        c="yellow",
        alpha=0.6,
      )
      plt.savefig(f"images/reconstruction/{tag}-view{i}.png", dpi=200)

    # shape parameters for ground truth
    b_gt = getShapeParameters(
      inputCoords["ALL"], target_lm, assam.model_s["ALL"], assam.std
    )
    shape_parameter_diff = b_gt - optAll["b"]
    print("parameter difference is")
    print(shape_parameter_diff)

    distX = utils.euclideanDist(
      outShape[:, [0]], ground_truth_landmarks[0][:, [0]]
    )
    dist2D = utils.euclideanDist(
      outShape[:, [0, 2]], ground_truth_landmarks[0][:, [0, 2]]
    )

    density_from_model = assam.getg_allModes(
      assam.density.mean(axis=0).reshape(-1),
      assam.model_g["ALL"][: len(optAll["b"])],
      optAll["b"] * np.sqrt(assam.variance),
    ).reshape(-1, config["training"]["num_imgs"])
    outShape_base = copy(outShape)
    ground_truth_landmarks_base = copy(ground_truth_landmarks[t])
    for i, (rotation_angle, axes) in enumerate(
      zip(config["training"]["rotation"], config["training"]["img_axes"])
    ):
      # plot density error between modelled gray-value and landmark gray-value
      density_error_for_point_cloud(
        utils.rotate_coords_about_z(outShape, rotation_angle),
        imgCoords,
        # img[i],
        target_img.reshape(-1, 500, 500)[i],
        density_from_model[:, i],
        tag=f"reconstructionFromModel_view{i}",
        axes=axes,
      )

      # get density error for true landmarks
      test_index = [i for i in range(0, len(lmIDs)) if lmIDs[i] == testID[t]]
      test_carina = nodalCoordsOrig[test_index[t], 1]

      # rotate ground truth dataset to get density for comparison
      img_centre = target_origin + (target_spacing * 500.0) / 2
      coords_rot = utils.rotate_coords_about_z(
        copy(ground_truth_landmarks_in_ct_space[t]),
        rotation_angle,
        img_centre,
      )
      img_coords_ground_truth = ssam.sam.drrArrToRealWorld(
        target_img.reshape(-1, 500, 500)[i], target_origin, target_spacing
      )[0]

      ground_truth_density = ssam.sam.getDensity(
        coords_rot,
        target_img.reshape(-1, 500, 500)[i],
        img_coords_ground_truth,
        axes,
      )

      # plot error between target gray-value and ground-truth density
      density_comparison(
        utils.rotate_coords_about_z(outShape, rotation_angle),
        coords_rot,
        target_img.reshape(-1, 500, 500)[i],
        imgCoords,
        density_at_lm_gt=ground_truth_density,
        tag=f"_view{i}",
        axes=axes,
      )

"""
# CHECK GROUND TRUTH OVERLAID ON XR
# get carina for alignment
test_index = [i for i in range(0,len(lmIDs)) 
                if lmIDs[i] == testID[t]]
test_carina = nodalCoordsOrig[test_index[t],1]

extent=[-img[0].shape[1]/2.*spacing_xr[0]+testOrigin[0][0],  
        img[0].shape[1]/2.*spacing_xr[0]+testOrigin[0][0],  
        -img[0].shape[0]*spacing_xr[2]+testOrigin[0][2],  
        testOrigin[0][2] ]

plt.close()
plt.imshow(img[0], cmap='gray', extent=extent)
plt.scatter(ground_truth_landmarks[0][:,0]+testOrigin[0][0], 
            ground_truth_landmarks[0][:,2]+testOrigin[0][2]+test_carina[2], 
            s=2, c='green')
plt.show()
"""
"""
ourShape = outShape - outShape[lmOrder['SKELETON'][1]]
projTest = copy(ground_truth_landmarks[0])
airwayProj_pts = lmOrder['Airway'][np.isin(lmOrder['Airway'], assam.projLM_ID['Airway'])]

dist2D_align = utils.euclideanDist(ourShape[:,[0,2]], projTest[:,[0,2]])
dist2D_align = dist2D_align[airwayProj_pts]
# plt.hist(dist2D_align, bins=100) ; plt.show()
plt.close()
plt.scatter(outShape[:,0][airwayProj_pts], outShape[:,2][airwayProj_pts], 
            s=2, c=dist2D_align)#c='blue')
plt.show()

# projTest -= projTest.mean(axis=0)
# projTest[:,2] -= lmAlign
plt.close()
# plt.imshow(img, cmap='gray', extent=extent)
plt.scatter(ourShape[:,0], ourShape[:,2], s=2, c='blue')
plt.scatter(projTest[:,0], projTest[:,2], s=2, c='black')
plt.show()
"""
"""

for i, (pID, lm, d) in enumerate(zip(patientIDs, landmarks, density)):
  plt.close()
  plt.scatter(lm[:,0], lm[:,2], 
              c=d[:,0], s=5, vmin=-2, vmax=2, cmap='gray')
  plt.title(pID)
  plt.axis('off')
  plt.savefig('images/SAM/alignmentCheck/frontal-{}.png'.format(pID))



airwayProj_pts = lmOrder['Airway'][np.isin(lmOrder['Airway'], assam.projLM_ID['Airway'])]
ourShape = (outShape - outShape[lmOrder['SKELETON'][1]])[airwayProj_pts]
projTest = copy(ground_truth_landmarks[0])[airwayProj_pts]
plt.close()
plt.scatter(ourShape[:,0], ourShape[:,2], s=2, c='blue')
plt.scatter(projTest[:,0], projTest[:,2], s=2, c='black')
plt.show()"""


"""

density_assam = ssam.sam.getDensity(assam.shape["ALL"], assam.img[0], 
  assam.imgCoords, [0,2])

density_assamimg_landmarks = ssam.sam.getDensity(landmarks[testSet[0]], 
    assam.img[0], assam.imgCoords, [0,2])

img_coords_ground_truth = ssam.sam.drrArrToRealWorld(img, origin[testSet[0]], target_spacing)[0]
d_truth = ssam.sam.getDensity(landmarks_in_ct_space[testSet[0]], drrArr[testSet[0]][0], img_coords_ground_truth, [0,2])


fig, ax = plt.subplots(1,3)
# ax[0].scatter(assam.shape["ALL"][:,0], assam.shape["ALL"][:,2],
#               c = density_assam, cmap="gray")
ax[0].scatter(landmarks[testSet[0]][:,0], landmarks[testSet[0]][:,2],
              c = density[testSet[0]][:, 0], cmap="gray")
ax[1].scatter(landmarks[testSet[0]][:,0], landmarks[testSet[0]][:,2],
              c = density_assamimg_landmarks, cmap="gray")
ax[2].scatter(landmarks[testSet[0]][:,0], landmarks[testSet[0]][:,2],
              c = d_truth, cmap="gray")
plt.show()

fig, ax = plt.subplots(1,3)
ax[0].scatter(landmarks[testSet[0]][:,0], landmarks[testSet[0]][:,2],
              c = density[testSet[0]][:, 0], cmap="gray")
ax[1].scatter(landmarks[testSet[0]][:,1], landmarks[testSet[0]][:,2],
              c = density[testSet[0]][:, 1], cmap="gray")
ax[2].scatter(
utils.rotate_coords_about_z(landmarks[testSet[0]], 180, landmarks[testSet[0]].mean(axis=0))[:,1],
                landmarks[testSet[0]][:,2],
              c = density[testSet[0]][:, 2], cmap="gray")
plt.show()
"""
