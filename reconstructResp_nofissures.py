"""
run script for reconstructing airways and lobes from an X-ray

if no input is given, default will create random test set of size = 1
"""

import argparse
import re
from copy import copy
from distutils.util import strtobool
from glob import glob
from os import makedirs
from sys import exit
from time import time

import hjson
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyssam
import vedo as v

import userUtils as utils
from respiratoryReconstructionSSAM import RespiratoryReconstructSSAM
from respiratorySSAM import RespiratorySSAM


def get_inputs():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config",
    "-cf",
    default="config_nofissures.json",
    type=str,
    help="input config file [default 'config_nofissures.json']",
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
    "--output_tag",
    "-o",
    default="reconstruction",
    type=str,
    help="output surface tag ",
  )
  parser.add_argument(
    "--described_variance",
    "-v",
    default=0.9,
    type=float,
    help="fraction of variance in training set to include in model [0,1]",
  )
  parser.add_argument(
    "--c_prior",
    "-cp",
    default=0.00044164808307051954,
    type=float,
    help="prior shape loss coefficient",
  )
  parser.add_argument(
    "--c_dense", "-cd", default=0.6869566761589552817, type=float, help="density loss coefficient"
  )
  parser.add_argument(
    "--c_edge",
    "-ce",
    default=0.7945582031917642896,
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
    "--debug",
    "-d",
    default=False,
    type=str,
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
  
  return parser.parse_args()


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
    "./images/reconstruction/density-error" + tag + ".png",
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
    "./images/reconstruction/density-comparison" + tag + ".png",
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
  input_landmarks_vec = input_landmarks.copy().reshape(-1)
  input_landmarks_vec -= input_landmarks_vec.mean()
  input_landmarks_vec /= input_landmarks_vec.std()
  average_landmarks_vec = average_landmarks.copy().reshape(-1)
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
    if plane == 1:
      plot_plane = 0
    else:
      plot_plane = 1
    key = "Airway"
    x = surfCoords_centred[key][projLM_ID[key]]
    plt.close()
    plt.scatter(x[:, plot_plane], x[:, 2], c="black", s=1)
    plt.scatter(
      landmarks["Airway"][:, plot_plane],
      landmarks["Airway"][:, 2],
      c="blue",
      s=10,
      alpha=0.3,
    )
    if args.debug == "s":
      plt.savefig(
        f"images/reconstruction/debug_newProjLMs_plane{plane}.png"
      )
    else:
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
  graph_dir_base,
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
  lgraphList = [nx.read_gpickle(graph_dir_base.replace("*", cID)) for cID in caseIDs]
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
  args = get_inputs()
  optimiser_kwargs = {}
  if not args.quiet:
    print(__doc__)
  startTime = time()

  if args.bayesian_opt:
    bayes_pts = np.loadtxt("points_to_sample.txt")
    c_dense = bayes_pts[0]
    c_edge = bayes_pts[1]
    c_prior = c_prior = (1.0e-4) ** (bayes_pts[2])
    optimiser_kwargs["c_edge_noisy_multiplier"] = bayes_pts[3]
  else:
    c_dense = args.c_dense
    c_edge = args.c_edge
    c_prior = args.c_prior
    optimiser_kwargs["c_edge_noisy_multiplier"] = 0.002
  print(f"c_dense is {c_dense}")
  print(f"c_edge is {c_edge}")
  print(f"c_prior is {c_prior}")
  c_anatomical = args.c_anatomical
  c_grad = args.c_grad
  kernel_radius = args.kernel_radius
  kernel_distance = args.kernel_distance

  shapes = ["Airway", "RUL", "RML", "RLL", "LUL", "LLL"]
  lobes = ["RUL", "RML", "RLL", "LUL", "LLL"]

  img = None
  spacing_xr = None

  print("\tReading data")
  with open(args.config) as f:
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
  landmark_files = filesFromRegex(config["luna16paths"]["landmarks"])
  lmIDs = matchesFromRegex(config["luna16paths"]["landmarks"])
  landmark_bifurcation_files = filesFromRegex(config["luna16paths"]["bifurcation_landmarks"])
  landmark_files.sort()
  landmark_bifurcation_files.sort()

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
    or len(landmark_files) == 0
    or len(spacingDirs) == 0
    or len(landmark_bifurcation_files) == 0
  ):
    raise AssertionError(
      "ERROR: The directories you have declared are empty.",
      "\nPlease check your input arguments.",
    )

  # remove scans without landmarks from DRR dirs
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

  landmarks = np.array(
    [np.loadtxt(l, delimiter=",", skiprows=1) for l in landmark_files]
  )
  nodalCoordsOrig = np.array(
    [
      np.loadtxt(l, delimiter=",", skiprows=1, usecols=[1, 2, 3])
      for l in landmark_bifurcation_files
    ]
  )

  lmOrder = dict.fromkeys(shapes)
  lmOrder["SKELETON"] = np.loadtxt(
    f"{config['luna16paths']['landmarks_index_dir']}/landmarkIndexSkeleton.txt"
  ).astype(int)
  lmOrder["LUNGS"] = []
  for shape in shapes:
    lmOrder[shape] = np.loadtxt(
      f"{config['luna16paths']['landmarks_index_dir']}/landmarkIndex{shape}.txt", dtype=int
    )
    if shape in lobes:
      lmOrder["LUNGS"].extend(list(lmOrder[shape]))
  lmOrder["LUNGS"] = np.array(lmOrder["LUNGS"])

  lgraph = getMeanGraph(
    patientIDs, 
    landmarks, 
    landmarks.mean(axis=0), 
    graph_dir_base=config["luna16paths"]["landmark_graphs"]
  )
  nx.write_gpickle(lgraph, "skelGraphs/nxGraphLandmarkMean.pickle")
  lgraph_branches = utils.simplifyGraph(lgraph)
  nx.write_gpickle(
    lgraph_branches, "skelGraphs/nxGraphLandmarkMeanBranchesOnly.pickle"
  )
  # read appearance modelling data
  origin = np.vstack([np.loadtxt(o, skiprows=1)] for o in originDirs)
  spacing = np.vstack(
    [np.loadtxt(o, skiprows=1)] for o in spacingDirs
  )
  # crop last two rows of pixels off XR so white pixels don't interfere with normalising
  drrArr = np.rollaxis(
    np.dstack(
      [utils.loadXR(o) for o in imDirs]
    ),
    2,
    0,
  )
  if "drrs_left" in config["training"]["img_keys"]:
    drrArr_left = np.rollaxis(
      np.dstack(
        [utils.loadXR(o) for o in imDirs_left]
      ),
      2,
      0,
    )
  if "drrs_right" in config["training"]["img_keys"]:
    drrArr_right = np.rollaxis(
      np.dstack(
        [
          utils.loadXR(o)
          for o in imDirs_right
        ]
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
  origin = copy(origin)
  spacing = copy(spacing)
  drrArr = copy(drrArr)

  # see if we use unseen image from different dataset, or use one from Luna16
  if "test_unseen_image" not in config["test-set"].keys():
    test_unseen_image = False
  else:
    test_unseen_image = bool(
      strtobool(config["test-set"]["test_unseen_image"][0])
    )

  if test_unseen_image:
    # get data for unseen test image
    testID = ["unseen"+args.case]
    test_img_files = [
      config["test-set"]["unseen_image"]["img_front"],
      config["test-set"]["unseen_image"]["img_left"],
    ]
    testIm = [np.rollaxis(
          np.dstack(
            [
              utils.loadXR(f)
              for f in test_img_files
            ]
          ),
          2,
          0,
        )]
    testOrigin = [np.loadtxt(
          config["test-set"]["unseen_image"]["origin"], skiprows=1
        )]
    testSpacing = [np.loadtxt(
          config["test-set"]["unseen_image"]["spacing"], skiprows=1
        )]

  else:
    # format data for testing by randomising selection and removing these
    # from training
    assignedTestIDs = [args.case]
    testSize = args.testSize
    testID = []
    randomise_testing = args.randomise  # False
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
      for i, pID in enumerate(patientIDs):
        if pID in assignedTestIDs:
          testSet.append(i)
    assert len(testSet) != 0, f"no cases in test set for patient {args.case}"

    testLM = []
    testOrigin = []
    testSpacing = []
    testIm = []
    testSet.sort()
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
    assert len(testID) != 0, f"no testID selected {args.case}, {assignedTestIDs}"

  meanArr = landmarks.mean(axis=0)

  # create shape and appearance model instance, based on pyssam base class
  ssam = RespiratorySSAM(
    landmarks,
    landmarks_in_ct_space,
    drrArr,
    origin,
    spacing,
    rotation=config["training"]["rotation"],
    img_coords_axes=config["training"]["img_axes"],
  )
  if testIm[0].ndim == 3:
    # if multiple images for reconstruction, get density for each one
    # which is last N columns from xg_train
    density = ssam.shape_appearance[:, :, -testIm[0].shape[0] :]
    number_of_features = 3 + testIm[0].shape[0]  # 3 = num coordinates
  else:
    density = ssam.shape_appearance[:, :, -1].reshape(len(landmarks), -1, 1)
    number_of_features = 4  # three coordinates, and a density value
  ssam.create_pca_model(ssam.shape_appearance_columns)
  model = ssam.pca_model_components

  meanArr = np.mean(landmarks, axis=0)

  # set number of modes
  numModes = np.where(
    np.cumsum(ssam.pca_object.explained_variance_ratio_) > args.described_variance
  )[0][0]
  if not args.quiet:
    print("modes used is", numModes)

  # center the lobes vertically
  # keep vertical alignment term for later use
  lmAlign = meanArr[:, 2].mean()  # landmarks[:,2].mean()
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
  # face normals of template mesh
  meanNorms_face = dict.fromkeys(shapes)  
  meanNorms_face_rot45 = dict.fromkeys(shapes)
  surfToLMorder = dict.fromkeys(shapes)  # mapping between surface nodes and LMs
  newMean = args.newMean
  # create mesh for population average from a morphing algorithm
  mean_shape_file = f"{config['luna16paths']['template_dir']}/meanAirway-orig.stl"

  for key in shapes:
    mean_shape_file = f"{config['luna16paths']['template_dir']}/mean{key}-orig.stl"
    assert (
      len(glob(mean_shape_file)) > 0
    ), f"file {mean_shape_file} does not exist!"

    mean_mesh[key] = v.load(mean_shape_file).computeNormals()

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
    else:
      mesh = mean_mesh[key].clone().decimate(fraction=0.1).clean()
    if not args.quiet:
      print("decimated num cells", len(mesh.faces()))
    mesh_45 = mesh.clone().rotateZ(45)
    meanNorms_face_rot45[key] = mesh_45.normals(cells=True)
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

  tagBase = copy(args.output_tag)
  for t, (
    target_id,
    target_img,
    _,#target_origin,
    target_spacing,
  ) in enumerate(zip(testID, testIm, testOrigin, testSpacing)):

    tag = tagBase + "_case" + target_id
    # load image directly from training data
    img = target_img.copy()

    # index 0 as output is stacked
    appearance_xr_helper = pyssam.utils.AppearanceFromXray(img, np.zeros(3), target_spacing)
    imgCoords = appearance_xr_helper.pixel_coordinates[0]
    get_density = appearance_xr_helper.compute_landmark_density
    assert appearance_xr_helper.pixel_coordinates.shape[0] == 1, f"unexpected shape for pixel_coordinates {imgCoords.shape}"
    spacing_xr = target_spacing.copy()
    # center image coords, so in the same coord system as edges
    imgCoords -= np.mean(imgCoords, axis=0)
    
    # edge points in units of pixels from edge map
    edgePoints = [None] * len(config["test-set"]["outlines"])
    for f, _ in enumerate(config["test-set"]["outlines"]):
      file_regex = config["test-set"]["outlines"][f].format(
        target_id, target_id
      )
      print(file_regex)
      edgePoints[f] = np.loadtxt(file_regex, delimiter=",")
      edgePoints[f] = np.unique(edgePoints[f], axis=0)

    if "outline_noisy" in config["test-set"]:
      file_regex = config["test-set"]["outline_noisy"].format(
        target_id, target_id
      )
      print(file_regex)
      edgePoints_noisy = np.loadtxt(file_regex, delimiter=",")
      edgePoints_noisy = np.unique(edgePoints_noisy, axis=0)
      optimiser_kwargs["outline_noisy"] = edgePoints_noisy

    # if only 1 x-ray given, change shape from list of 2D arrays to one 2D array
    if len(edgePoints) == 1:
      edgePoints = edgePoints[f]
    if config["training"]["num_imgs"] >= 2:
      optimiser_kwargs["bounds_index_pose"] = [0, 1, 2]
      optimiser_kwargs["bounds_index_scale"] = 3
      optimiser_kwargs["bounds_index_shape"] = 4
    else:
      optimiser_kwargs["bounds_index_pose"] = [0, 1]
      optimiser_kwargs["bounds_index_scale"] = 2
      optimiser_kwargs["bounds_index_shape"] = 3


    # declare posterior shape model class
    assam = RespiratoryReconstructSSAM(
      ssam_obj=ssam,
      shape=inputCoords,
      xRay=edgePoints,
      lmOrder=lmOrder,
      normals=None,
      transform=carinaArr[t] * 0.0,
      img=copy(img),
      imgCoords=imgCoords,
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
      **optimiser_kwargs,
    )

    assam.spacing_xr = spacing_xr
    # import functions to PSM class
    assam.get_density = get_density

    print("getting projected landmarks")
    projLM_file = config["luna16paths"]["projLM_ID_dir"] + "/projectedMeanLandmarks{}.csv"
    projLM_ID_file = config["luna16paths"]["projLM_ID_dir"] + "/projectedMeanLandmarksID{}.csv"
    t1 = time()

    # True if given new mesh to get projection landmarks.
    if len(glob(projLM_ID_file.format("*"))) == 0 or args.newProjLM:
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
            if args.debug:
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
          if args.debug:
            print(config["luna16paths"]["projLM_ID_file"].format(key, proj_ind))
          assam.projLM_ID_multipleproj[proj_ind][key] = np.loadtxt(
            config["luna16paths"]["projLM_ID_file"].format(key, proj_ind),
            dtype=int,
            skiprows=1,
          )
        if args.debug:
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

    # assam.projLM_IDAll = []
    # pointCounter = 0
    # for key in assam.projLM_ID.keys():
    #   if key not in ["Airway", "RML"]:
    #     assam.projLM_IDAll.extend(
    #       list(np.array(assam.projLM_ID[key]) + pointCounter)
    #     )
    #     # tmp_id = np.arange(0, inputCoords[key].shape[0], 1)
    #     # assam.projLM_IDAll.extend(tmp_id+pointCounter)
    #   pointCounter += inputCoords[key].shape[0]
    # assam.projLM_IDAll = np.array(assam.projLM_IDAll)

    # initialise parameters to be optimised - including initial values + bounds
    optTrans_new = dict.fromkeys(["pose", "scale"])
    if config["training"]["num_imgs"] >= 2:
      optTrans_new["pose"] = [0, 0, 0]
    else:
      optTrans_new["pose"] = [0, 0]
    optTrans_new["scale"] = 1
    # estimates on roughly how small/large the lungs can deviate from the mean
    MIN_SCALE = 0.4
    MAX_SCALE = 2
    # estimate of roughly how far off-centre the lungs can move
    MAX_TRANSLATION = 20
    # from literature, SSM parameters should be no more than 3x the model variance
    SSAM_PARAM_RANGE = 3 
    if config["training"]["num_imgs"] >= 2:
      initPose = np.array([optTrans_new["pose"][0], optTrans_new["pose"][1], optTrans_new["pose"][2], optTrans_new["scale"]])
      bounds = np.array(
        [
          (-MAX_TRANSLATION, MAX_TRANSLATION),
          (-MAX_TRANSLATION, MAX_TRANSLATION),
          (-MAX_TRANSLATION, MAX_TRANSLATION),
          (MIN_SCALE, MAX_SCALE),
          (-SSAM_PARAM_RANGE, SSAM_PARAM_RANGE),
        ]
      )
    else:
      initPose = np.array([optTrans_new["pose"][0], optTrans_new["pose"][1], optTrans_new["scale"]])
      bounds = np.array(
        [
          (-MAX_TRANSLATION, MAX_TRANSLATION),
          (-MAX_TRANSLATION, MAX_TRANSLATION),
          (MIN_SCALE, MAX_SCALE),
          (-SSAM_PARAM_RANGE, SSAM_PARAM_RANGE),
        ]
      )

    # initialise parameters that control optimisation process
    assam.optIterSuc, assam.optIter = 0, 0
    assam.scale = optTrans_new["scale"]
    assert len(assam.projLM_ID) != 0, "no projected landmarks"

    # for debugging input data, run only up to just before optimisation starts
    if args.load_data_only:
      exit()
    makedirs("images/reconstruction/debug", exist_ok=True)

    # perform optimisation
    optAll = assam.optimiseAirwayPoseAndShape(
      assam.loss_function, initPose, bounds, epochs=args.epochs, threads=1
    )
    print(
      "\n\n\n\n\n\t\t\tTime taken is {0} ({1} mins)".format(
        time() - startTime, round((time() - startTime) / 60.0), 3
      )
    )

    # get final shape
    output_shape_morphed = assam.morphAirway(
      inputCoords["ALL"],
      optAll["b"],
      assam.model_s["ALL"][: len(optAll["b"])],
    )
    outShape = assam.centerThenScale(
      output_shape_morphed, optAll["scale"], output_shape_morphed.mean(axis=0)
    )
    outShape += optAll["pose"]

    out_file = "{}_{}.{}"
    makedirs("outputLandmarks", exist_ok=True)
    out_lm_file = "outputLandmarks/" + out_file
    np.savetxt(
      out_lm_file.format(tag, "ALL", "csv"),
      outShape,
      header="x, y, z",
      delimiter=",",
    )

    if config["training"]["num_imgs"] == 1:
      axes = config["training"]["img_axes"][0]
      extent = [
        imgCoords[:, axes[0]].min(),
        imgCoords[:, axes[0]].max(),
        imgCoords[:, axes[1]].min(),
        imgCoords[:, axes[1]].max(),
      ]
      rotation_angle = config["training"]["rotation"][0]
      coords_rot = utils.rotate_coords_about_z(copy(outShape), rotation_angle)
      plt.close()
      plt.imshow(img, cmap="gray", extent=extent)
      plt.scatter(edgePoints[:, 0], edgePoints[:, 1], s=2, c="black")
      plt.scatter(
        coords_rot[:, axes[0]],
        coords_rot[:, axes[1]],
        s=2,
        c="yellow",
        alpha=0.6,
      )
      plt.savefig(f"images/reconstruction/{tag}-view{0}.png", dpi=200)
    else:
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
        plt.scatter(edgePoints[i][:, 0], edgePoints[i][:, 1], s=2, c="black")
        plt.scatter(
          coords_rot[:, axes[0]],
          coords_rot[:, axes[1]],
          s=2,
          c="yellow",
          alpha=0.6,
        )
        plt.savefig(f"images/reconstruction/{tag}-view{i}.png", dpi=200)

    # we do not have landmarks etc so we cannot plot comparisons
    if test_unseen_image:
      continue
    else:
      target_lm = testLM[t]

    # shape parameters for ground truth
    b_gt = getShapeParameters(
      inputCoords["ALL"], target_lm, assam.model_s["ALL"], assam.std
    )
    shape_parameter_diff = b_gt - optAll["b"]
    print("parameter difference is")
    print(shape_parameter_diff)
