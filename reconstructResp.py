'''
    run script for reconstructing airways amd lobes from an X-ray
'''

from concurrent import futures
import userUtils as utils
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
from os import remove

import vedo as v
from vedo import printc
import nevergrad as ng

from math import pi
import argparse
from copy import copy
from sys import exit, argv
from glob import glob
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler

from skimage import io
from skimage.color import rgb2gray

from time import time
from datetime import date


''' TODO - sync SAM and SSM classes to conda folder. Can then oimport like
from ssm import SSM as RespiratorySSM 
'''
# from reconstructSSAM import LobarPSM
from respiratorySAM import RespiratorySAM
from respiratorySSM import RespiratorySSM
from respiratorySSAM import RespiratorySSAM
from respiratoryReconstructionSSAM import RespiratoryReconstructSSAM
from morphAirwayTemplateMesh import MorphAirwayTemplateMesh
from morphTemplateMesh import MorphTemplateMesh as MorphLobarTemplateMesh

tag = "_case0"
template_mesh = "segmentations/template3948/newtemplate3948_mm.stl"
plotAlignment = False

def getInputs():
  parser = argparse.ArgumentParser(description='SSM for lung lobe variation')
  parser.add_argument('--inp', '-i',
                      default='allLandmarks/', 
                      type=str, 
                      help='input files (landmarks)'
                      )
  parser.add_argument('--case', '-c',
                      default='8684',#'3948', 
                      type=str,#, required=True,
                      help='training data case'
                      )
  parser.add_argument('--out', '-o',
                      default='reconstruction', 
                      type=str,
                      help='output surface tag '
                      )
  parser.add_argument('--var', '-v',
                      default=0.9, 
                      type=float, 
                      help='fraction of variance in training set to include in model [0,1]'
                      )
  parser.add_argument('--c_prior', '-cp',
                      default=0.025, 
                      type=float, 
                      help='prior shape loss coefficient'
                      )
  parser.add_argument('--c_dense', '-cd',
                      default=1., 
                      type=float, 
                      help='density loss coefficient'
                      )
  parser.add_argument('--c_edge', '-ce',
                      default=0.01, 
                      type=float, 
                      help='edge map loss coefficient'
                      )
  parser.add_argument('--c_anatomical', '-ca',
                      default=0.6, 
                      type=float, 
                      help='anatomical shadow loss coefficient'
                      )
  parser.add_argument('--c_grad', '-cg',
                      default=0.4, 
                      type=float, 
                      help='image gradient loss coefficient'
                      )
  parser.add_argument('--kernel_radius', '-kr',
                      default=9, 
                      type=int, 
                      help='radius (pixels) of image kernels'
                      )
  parser.add_argument('--kernel_distance', '-kd',
                      default=18, 
                      type=int, 
                      help='distance (pixels) between image kernels'
                      )
  parser.add_argument('--drrs', 
                      default='../xRaySegmentation/DRRs_enhanceAirway/luna16_cannyOutline/', 
                      type=str, 
                      help='input files (drr)'
                      )
  parser.add_argument('--meshdir', '-m',
                      default='segmentations/', 
                      type=str, 
                      help='directory of surface files'
                      )
  parser.add_argument('--shapes', 
                      default="*", 
                      type=str, 
                      help='which shape would the user like to grow?'+\
                            'Corresponds to string common in landmarks text files'+\
                            '\nRUL, RML, RLL, LUL, LLL, or ALL?'
                      )
  parser.add_argument('--debug', '-d',
                      default=False, 
                      type=bool, 
                      help='debug mode -- shows print checks and blocks'+
                            'plotting outputs'
                      )
  parser.add_argument('--epochs', '-e',
                      default=4000, 
                      type=int, 
                      help='number of optimisation iterations'
                      )
  parser.add_argument('--xray', '-x',
                      default=False, 
                      type=str,# required=True,
                      help='X-ray outline to use for fitting (2xN csv)'
                      )
  parser.add_argument('--imgSpacing', 
                      default=1, 
                      type=int, 
                      help='multiplier to coarsen images (must be int)'
                      )

  args = parser.parse_args()
  # inputDir = args.inp
  # tag = args.out
  # case = args.case
  # var = args.var
  # drrDir = args.drrs
  # debugMode = args.debug 
  # shapeKey = args.shapes.split()
  # surfDir = args.meshdir
  # numEpochs = args.epochs
  # xray = args.xray
  # c_edge = args.c_edge
  # c_dense = args.c_dense
  # c_prior = args.c_prior
  # imgSpacing = args.imgSpacing

  # return inputDir, case, tag, var, drrDir, \
  #         debugMode, shapeKey, surfDir, numEpochs, \
  #         xray, c_edge, c_dense, c_prior, imgSpacing
  return args

def getShapeParameters(average_landmarks, input_landmarks, 
                       shape_model, model_std):
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

  shape_parameters = np.dot((input_landmarks_vec-average_landmarks_vec), 
                              shape_model[:len(model_std)].T)/model_std
  return shape_parameters

def getMeanGraph(caseIDs, 
                landmarks,
                mean_landmarks,
                graph_files='landmarks/manual-jw-diameterFromSurface/nxGraph{}landmarks.pickle'):
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
      pos.append(lgraph.nodes[node]['pos'])
    pos = np.vstack(pos)
    posList.append(pos)
  lgraphMean = lgraphList[-1].copy()
  lgraph = lgraphList[-1].copy()
  for node in lgraph.nodes:
    pos = lgraph.nodes[node]['pos']
    dist = utils.euclideanDist(landmarks[i], pos)
    currentLM = np.argmin(dist)
    # find closest graph node, if it is not a landmark then find next closest
    isin = np.isclose(posList[i], landmarks[i][currentLM]).all(axis=1)
    while isin.sum() == 0:
      dist[currentLM] = 100000000
      currentLM = np.argmin(dist)
      isin = np.isclose(posList[i], landmarks[i][currentLM]).all(axis=1)
    # assign metadata to graphs
    lgraph.nodes[node]['npID'] = currentLM
    lgraphMean.nodes[node]['pos'] = mean_landmarks[currentLM]
    lgraphMean.nodes[node]['npID'] = currentLM
  return lgraphMean

if __name__=='__main__':
  date_today = str(date.today())
  print(__doc__)
  startTime = time()

  args = getInputs()
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
  c_edge = args.c_edge
  c_dense = args.c_dense
  c_prior = args.c_prior
  c_anatomical = args.c_anatomical
  c_grad = args.c_grad
  kernel_radius = args.kernel_radius
  kernel_distance = args.kernel_distance
  imgSpaceCoeff = args.imgSpacing


  # landmarkDir, case, tag, describedVariance, drrDir, debug, \
  #         shapes, surfDir, numEpochs, xrayEdgeFile, \
  #         c_edge, c_dense, c_prior, imgSpaceCoeff = getInputs()
  img=None
  spacing_xr=None

  print("\tReading data")
  #-read DRR data
  originDirs = glob( drrDir + "/origins/origins/drr*.md")#.sort()
  spacingDirs = glob( drrDir + "/*/drr*.md")#.sort()
  imDirs = glob(drrDir + "/*/drr*.png")#.sort()
  originDirs.sort()
  spacingDirs.sort()
  imDirs.sort()

  patientIDs = [i.split("/")[-1].replace(".png", "")[-4:] for i in imDirs]
  landmarkDirs = glob( landmarkDir+"/allLandmarks*.csv" )
  lmIDs = [i.split("/")[-1].split("andmarks")[1][:4] for i in landmarkDirs]
  landmarkDirsOrig = glob('landmarks/manual-jw/landmarks*.csv')
  landmarkDirs.sort()
  landmarkDirsOrig.sort()
  if len(imDirs) == 0 \
  or len(originDirs) == 0 \
  or len(landmarkDirs) == 0 \
  or len(spacingDirs) == 0 \
  or len(landmarkDirsOrig) == 0: 
    print("ERROR: The directories you have declared are empty.",
          "\nPlease check your input arguments.")
    exit()
  # transDirs = glob( "savedPointClouds/allLandmarks/"
  #                             +"transformParams_case*"
  #                             +"_m_"+shape+".dat")
  # transDirs_all = glob( "savedPointClouds/allLandmarks/"
  #                             +"transformParams_case*"
  #                             +"_m_"+shape+".dat")
  #-remove scans without landmarks from DRR dirs
  missing = []
  missingID = []
  
  delInd = []
  for i, imD in enumerate(imDirs):
    currID = imD.split('.')[-2][-4:]
    if currID not in lmIDs:
      delInd.append(i)

  for dId in delInd[::-1]:
    originDirs.pop(dId)
    spacingDirs.pop(dId)
    imDirs.pop(dId)
    patientIDs.pop(dId)
  missing = []
  missingID = []
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

  landmarks = np.array([np.loadtxt(l, delimiter=",",skiprows=1) 
                            for l in landmarkDirs])
  nodalCoordsOrig = np.array([np.loadtxt(l, delimiter=",",skiprows=1,usecols=[1,2,3]) 
                            for l in landmarkDirsOrig])

  lmOrder = dict.fromkeys(shapes)
  lmOrder['SKELETON'] = np.loadtxt(landmarkDir+'landmarkIndexSkeleton.txt',
                                    dtype=int)
  lgraph = getMeanGraph(patientIDs, landmarks, landmarks.mean(axis=0))
  nx.write_gpickle(lgraph, 'skelGraphs/nxGraphLandmarkMean.pickle')
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

  #-read appearance modelling data
  origin = np.vstack([np.loadtxt(o, skiprows=1)] for o in originDirs)
  spacing = np.vstack([np.loadtxt(o, skiprows=1)*imgSpaceCoeff] 
                              for o in spacingDirs)
  # crop last two rows of pixels off XR so white pixels don't interfere with normalising
  drrArr = np.rollaxis(
                      np.dstack([utils.loadXR(o)[:-2,:-2][::imgSpaceCoeff,::imgSpaceCoeff] 
                        for o in imDirs]),
                      2, 0)
  carinaArr = nodalCoordsOrig[:,1]
  #-offset centered coordinates to same reference frame as CT data
  lmProj = landmarks + carinaArr[:,np.newaxis]

  #-load pre-prepared mean stl and point cloud of all data
  meanArr = landmarks.mean(axis=0)

  #-reset new params
  landmarksAll=0
  origin = copy(origin)
  spacing = copy(spacing)
  drrArr = copy(drrArr)
  # landmarks = lmProj.copy()
      
  #-format data for testing by randomising selection and removing these 
  # from training
  assignedTestIDs = ["0645", "3948", "5268", "6730", "8865"]
  assignedTestIDs = ["3948"]
  assignedTestIDs = [case]
  testSize = 1
  testID = []
  randomise_testing =  False
  if randomise_testing:
    testSet = np.random.randint(0,len(patientIDs)-1, testSize)
    # check if test and training datasets share overlapping samples
    testOverlapTrain = True  
    while np.unique(testSet).size != testSize and testOverlapTrain:
      testSet = np.random.randint(0,len(patientIDs)-1, testSize)
      testOverlapTrain = [True for p in testID if p in assignedTestIDs]

      for t in testSet[::-1]:
        #-store test data in different list
        testID.append(patientIDs[t])
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
  lmProjDef = lmProj.copy()
  landmarksDef = landmarks.copy()
  lmProj_test = []
  for t in testSet[::-1]:
    #-store test data in different list
    testID.append(patientIDs[t])
    testOrigin.append(origin[t])
    testSpacing.append(spacing[t])
    testIm.append(drrArr[t])
    testLM.append(copy(landmarks[t]))
    lmProj_test.append(copy(landmarks[t]))
    #-remove test data from train data
    patientIDs.pop(t)
    origin = np.delete(origin,t,axis=0)
    spacing = np.delete(spacing,t,axis=0)
    drrArr = np.delete(drrArr,t,axis=0)
    landmarks = np.delete(landmarks,t,axis=0)
    lmProj = np.delete(lmProj,t,axis=0)

  meanArr = landmarks.mean(axis=0)

  # template_lm_file_lobes = landmarkDir+"/landmarks0{}-case8684.csv"

  #-create appearance model instance and load data
  ssam = RespiratorySSAM(landmarks, 
                          lmProj, 
                          drrArr, 
                          origin,
                          spacing,
                          train_size=landmarks.shape[0])
  density  = ssam.xg_train[:,:,-1]
  model = ssam.phi_sg
  meanArr = np.mean(landmarks, axis=0)

  #-set number of modes
  numModes = np.where(
                      np.cumsum(ssam.pca_sg.explained_variance_ratio_)
                      >describedVariance)[0][0]
  print("modes used is", numModes)

  #-center the lobes vertically
  #-keep vertical alignment term for later use
  lmAlign = meanArr[:,2].mean() #landmarks[:,2].mean()
  meanArr[:,2] -= lmAlign

  shapes = ['Airway', 'RUL', 'RML', 'RLL', 'LUL', 'LLL']
  # shapes = ['Airway']
  lobes = ['RUL', 'RML', 'RLL', 'LUL', 'LLL']
  # numbering for each lobe in file
  lNums = {"RUL": "4",
           "RML": "5",
           "RLL": "6",
           "LUL": "7",
           "LLL": "8"
           }
  
  modelDict = dict.fromkeys(shapes)
  inputCoords = dict.fromkeys(shapes)
  inputCoords['ALL'] = meanArr
  modelDict['ALL'] = model
  for shape in shapes:
    lmOrder[shape] = np.loadtxt(landmarkDir+'landmarkIndex{}.txt'.format(shape),
                                dtype=int)
    inputCoords[shape] = meanArr[lmOrder[shape]]
    modelDict[shape] = model.reshape(len(landmarks), -1, 4)[:,lmOrder[shape]]

  mean_mesh = dict.fromkeys(shapes)
  faces = dict.fromkeys(shapes) # faces of mean surface for each shape
  surfCoords_centred = dict.fromkeys(shapes)
  surfCoords_mmOrig = dict.fromkeys(shapes) # surface nodes for each shape
  surfCoords_mm = dict.fromkeys(shapes) # surface nodes in same ordering as LMs
  meanNorms_face = dict.fromkeys(shapes) # normals for each face (?) of mean mesh
  surfToLMorder = dict.fromkeys(shapes) # mapping between surface nodes and LMs
  newMean = False
  # create mesh for population average from a morphing algorithm
  templateDir = 'templates/coarserTemplates/'
  mean_shape_file = templateDir+'meanAirway.stl'

  assert  glob(mean_shape_file) != 0 or not newMean, 'error in loading meshes. We created coarsened ones manually - do not overwrite'
  # if glob(mean_shape_file) == 0 or newMean:
  #   print('making airway template mesh')
  #   template_lmFileOrig = 'landmarks/manual-jw/landmarks3948.csv'
  #   # template_lmFile = 'landmarks/manual-jw-diameterFromSurface/landmarks3948_diameterFromSurf.csv'
  #   template_lmFile = 'allLandmarks/allLandmarks3948.csv'
  #   template_airway_file = 'segmentations/template3948/newtemplate3948_mm.stl'
  #   lm_template = np.loadtxt(template_lmFile, skiprows=1, delimiter=",",
  #                               usecols=[0,1,2])
  #   carinaTemplate = np.loadtxt(template_lmFileOrig, skiprows=1, delimiter=',',
  #                               usecols=[1,2,3])[1]*-1
  #   template_airway_mesh = v.load(template_airway_file)
  #   template_airway_mesh = template_airway_mesh.pos(carinaTemplate)
  #   morph_airway = MorphAirwayTemplateMesh(lm_template[lmOrder['Airway']], 
  #                                           meanArr[lmOrder['Airway']], 
  #                                           template_airway_mesh,
  #                                           sigma=0.3,
  #                                           quiet=True)
  #   morph_airway.mesh_target.write(mean_shape_file)
  #   np.savetxt('templates/meanAirway.csv', meanArr[lmOrder['Airway']],
  #               delimiter=',', header='x, y, z')
  #   for lobe in lobes:
  #     print('making {} template mesh'.format(lobe))
  #     template_lmFile = 'allLandmarks/allLandmarks8684.csv'
  #     lm_template = np.loadtxt(template_lmFile, skiprows=1, delimiter=",",
  #                                 usecols=[0,1,2])
  #     template_mesh_file_lobes = "/home/josh/3DSlicer/project/luna16Rescaled/case8684/8684_mm_{}.stl"
  #     # template_mesh_file_lobes = "/home/josh/3DSlicer/project/luna16Rescaled/case3948/3948_mm_{}.stl"
  #     # lm_template_lobes = np.loadtxt(template_lm_file_lobes.format(key), delimiter=",")
  #     template_lobe_mesh = v.load(template_mesh_file_lobes.format(lNums[lobe]))
  #     morph_lobe = MorphLobarTemplateMesh(lm_template[lmOrder[lobe]], 
  #                                         meanArr[lmOrder[lobe]], 
  #                                         template_lobe_mesh,
  #                                         sigma=0.3,
  #                                         quiet=True)
  #     mean_lobe_file_out = templateDir+'mean{}.stl'.format(lobe)
  #     morph_lobe.mesh_target.write(mean_lobe_file_out)
  #     np.savetxt('templates/mean{}.csv'.format(lobe), meanArr[lmOrder[lobe]],
  #                 delimiter=',', header='x, y, z')

  for key in shapes:
    mean_shape_file = templateDir+'mean{}.stl'.format(key)
    assert len(glob(mean_shape_file))>0, 'file {} does not exist!'.format(mean_shape_file)
    mean_mesh[key] = v.load(mean_shape_file).computeNormals()
    # mesh_template_lms[key] =

  # exit()
  #-reorder unstructured stl file to be coherent w/ model and landmarks
  #-extract mesh data (coords, normals and faces)
  for key in shapes:
    print('loading {} mesh'.format(key))
    print('original num cells', len(mean_mesh[key].faces()))
    if key == 'Airway':
      mesh = mean_mesh[key].clone()
      pass
      # mesh = mean_mesh[key].clone().decimate(N=40e3).clean()
    else:
      mesh = mean_mesh[key].clone().decimate(fraction=0.1).clean()
    print('decimated num cells', len(mesh.faces()))
    #-load mesh data and create silhouette
    surfCoords = mesh.points()
    meanNorms_face[key] = mesh.normals(cells=True)
    faces[key] = np.array(mesh.faces())

    #-offset to ensure shapes are aligned to carina
    surfCoords_centred[key] = copy(surfCoords)
    surfCoords_mm[key] = surfCoords + carinaArr.mean(axis=0)
    surfCoords_mmOrig[key] = copy(surfCoords_mm[key])
    surfToLMorder[key] = []
    for point in meanArr[lmOrder[key]]:
      surfToLMorder[key].append( np.argmin( utils.euclideanDist(surfCoords, 
                                                                    point) ) )
    surfCoords_mm[key] = surfCoords_mm[key][surfToLMorder[key]]

  tagBase = copy(tag)
  for t, (tID, tLM, tImg, tOrig, tSpace) \
    in enumerate(zip(testID, testLM, testIm, testOrigin, testSpacing)):

    tag = tagBase+"_case"+tID
    #-declare test image and pre-process
    # imgOrig = copy(img) #-backup test image original
    img = tImg.copy() #ssam.imgsN[caseIndex] #-load image directly from training data
    img = ssam.sam.normaliseTestImageDensity(img) #-normalise "unseen" image
    imgCoords = ssam.sam.drrArrToRealWorld(img,
                                           np.zeros(3), 
                                           tSpace# [spacing_xr[0]]*3
                                          )[0]#-index 0 as output is stacked
    spacing_xr = tSpace.copy() #*imgSpaceCoeff #-no need to multiply as already done earlier
    #-center image coords, so in the same coord system as edges
    imgCoords -= np.mean(imgCoords, axis=0)#np.array([250,250])#*spacing_xr[[0,2]]

    #-for plotting image in same ref frame as the edges
    extent=[-img.shape[1]/2.*spacing_xr[0],  
            img.shape[1]/2.*spacing_xr[0],  
            -img.shape[0]/2.*spacing_xr[2],  
            img.shape[0]/2.*spacing_xr[2] ]
    extent_tmp = np.array(extent)

    #-remove old optimisation visualisations
    # remFiles = glob("images/xRayRecon/nevergrad/*.png")
    # for file in remFiles:
    #     remove(file)
    #-edge points in units of pixels from edge map
    xrayEdgeFile = "{}/{}/drr-outline-{}.csv".format(drrDir, tID, tID)
    edgePoints = np.loadtxt(xrayEdgeFile,
                            delimiter=",")
    # edgePoints_tmp = edgePoints#*512/500
    #-edge points in units of mm
    #-for some reason spacing_xr[[0,1]] gives correct height of edge map?
    edgePoints_mm = edgePoints 
    edgePoints_mm = np.unique(edgePoints_mm, axis=0)
    print("check height", edgePoints_mm[:,1].max() - edgePoints_mm[:,1].min())
    print("check width", edgePoints_mm[:,0].max() - edgePoints_mm[:,0].min())
    if debug:
      fig, ax = plt.subplots(2)
      ax[0].imshow(img,cmap="gray", extent=extent) 
      ax[0].scatter(edgePoints[:,0], edgePoints[:,1],s=2) 
      ax[1].scatter(meanArr[:,0], meanArr[:,2],s=1, c="black") 
      ax[1].scatter(edgePoints[:,0], edgePoints[:,1],s=2, c="blue") 
      plt.show()

    # # fig, ax = plt.subplots(2)
    # # ax[1].scatter(meanArr[:,0], meanArr[:,2],s=1, c="black") 
    # # ax[1].scatter(edgePoints[:,0], edgePoints[:,1],s=2, c="blue") 
    # plt.imshow(img,cmap="gray", extent=extent) 
    # plt.scatter(lmProjDef[10][:,0], 
    #             lmProjDef[10][:,2]-lmProjDef[10,:,2].mean(), 
    #             s=10) 
    # plt.show()

    # inputCoords = meanArr.copy()
    ##########################################################################
    #-declare posterior shape model class
    assam = RespiratoryReconstructSSAM(shape=inputCoords,
                                      xRay=edgePoints_mm,
                                      lmOrder=lmOrder,
                                      normals=None, 
                                      transform=carinaArr[t]*0.,
                                      img=img,
                                      # imgSpacing=spacing_xr,
                                      imgCoords=imgCoords,
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
                                      )
    assam.spacing_xr = spacing_xr
    #-import variables to class
    assam.variance = ssam.variance[:numModes]
    assam.std = ssam.std[:numModes]
    #-import functions to PSM class
    assam.getg_allModes = ssam.sam.getg_allModes
    assam.getDensity = ssam.sam.getDensity
    assam.normaliseTestImageDensity = ssam.sam.normaliseTestImageDensity
    # exit()
    #############################################################################

    print('getting projected landmarks')
    projLM_file = 'allLandmarks/projectedMeanLandmarks{}.csv'
    projLM_ID_file = 'allLandmarks/projectedMeanLandmarksID{}.csv'
    t1 = time()
    new_projection = False # True if given new mesh to get projection landmarks.
    if len(glob(projLM_file.format('*'))) == 0 or new_projection:
      assam.projLM, assam.projLM_ID = assam.getProjectionLandmarks(faces, 
                                                                  meanNorms_face, 
                                                                  surfCoords_mmOrig)
      assam.projLM, assam.projLM_ID = assam.deleteShadowedEdges(surfCoords_mm, 
                                                                assam.projLM, 
                                                                assam.projLM_ID,
                                                                )
      #-reorder projected surface points to same order as landmarks
      print('number proj airway pts', len(assam.projLM_ID['Airway']))
      for key in shapes:
        print('reordering projected landmarks for', key)
        delInd = []
        if key == 'Airway':
          # for p, pointID in enumerate(assam.projLM_ID[key]):
          newProjIDs = []
          newProjPos = []
          for p, point in enumerate(surfCoords_centred[key][assam.projLM_ID[key]]):
            dist = utils.euclideanDist(inputCoords[key], point)
            closest_lm_index = np.argmin(dist)
            if closest_lm_index not in newProjIDs:
              newProjIDs.append(closest_lm_index)
              newProjPos.append(inputCoords[key][closest_lm_index,[0,2]])
          assam.projLM_ID[key] = copy(newProjIDs)
          assam.projLM[key] = copy(np.vstack(newProjPos))
        else:
          for p, point in enumerate(assam.projLM_ID[key]):
            if np.isin(surfToLMorder[key], point).sum() > 0: #np.isin(point, surfToLMorder):
              mappedLM = np.argwhere(np.isin(surfToLMorder[key], point))
              assam.projLM_ID[key][p] = mappedLM[0][0]
            else:
              delInd.append(p)
          # print('finished reordering projected landmarks')
          #-delete projected surfPoints which were not included in mapping to LM space
          assam.projLM_ID[key] = np.delete(assam.projLM_ID[key], delInd)
          assam.projLM[key] = np.delete(assam.projLM[key], delInd)

        np.savetxt(projLM_file.format(key), assam.projLM[key], 
                    header='x y', delimiter=',')
        np.savetxt(projLM_ID_file.format(key), assam.projLM_ID[key], 
                    header='ID', fmt='%i')
    else:
      assam.projLM, assam.projLM_ID = dict.fromkeys(shapes), dict.fromkeys(shapes)
      for key in shapes:
        assam.projLM[key] = np.loadtxt(projLM_file.format(key), 
                                       skiprows=1, delimiter=',')
        assam.projLM_ID[key] = np.loadtxt(projLM_ID_file.format(key), 
                                          dtype=int, skiprows=1)

    print('time taken to get projected points', round(time()-t1),'s')

    # ids = np.arange(0, len(points))
    # norms = np.where( np.isin(faceIDs[:,0], ids) 
    #                           | np.isin(faceIDs[:,1], ids) 
    #                           | np.isin(faceIDs[:,2], ids)
    #                         )[0]

    # for i in ids:
    #   tmp = np.where(faceIDs[:,0]==i)
    #   if len(tmp[0])>1:
    #     print(tmp)
    # # norms = np.where( np.isin(ids, faceIDs[:,0]) 
    # #                           | np.isin(ids, faceIDs[:,1]) 
    # #                           | np.isin(ids, faceIDs[:,2])
    # #                         )[0]

    # # if len(norms[pID]) > 1:
    # '''check if y normal for point has +ve and -ve components
    #    in the projection plane '''
    # ynorms = faceNorms['RUL'][:,1]
    # if np.min(ynorms[norms[pID]]) < 0 \
    #   and np.max(ynorms[norms[pID]]) > 0:
    # projectionLM_ID = 
    print('finished getting projected landmarks. Time taken = {} s'.format(time()-t1))

    assam.fissureLM_ID = 0

    scaleInit = np.array([1,1])
    initPose = np.array([
                          0,
                          0,
                          scaleInit[0], # scale in horizontal dir
                        ])

    bounds = np.array([
                      (-np.inf, np.inf),
                      (-np.inf, np.inf),
                      (0.7, 2),
                      (-3,3)])#(-3, 3)])

    t1 = time()

    # lobeBackup = copy(assam)
    # id_backup = copy(assam.projLM_ID)

    if debug:
      plot_pts = surfCoords_mm['Airway'][assam.projLM_ID['Airway']]
      print('number proj airway pts', len(assam.projLM_ID['Airway']))
      plt.close()
      plt.scatter(plot_pts[:,0], plot_pts[:,2])
      plt.show()
      # plt.scatter(assam.projLM['Airway'][:,0], assam.projLM['Airway'][:,1])
      # plt.show()
    '''
    plot_pts = surfCoords_mm['Airway'][assam.projLM_ID['Airway']]

    plt.close()
    plt.scatter(plot_pts[:,0], plot_pts[:,2])
    plt.show()
    '''
    #-map projected landmark IDs for each lobe to their correpsonding position in
    #-the 'all' landmark array
    # assam.projLM_ID = []
    #######################################################################
    optTrans_new = dict.fromkeys(["pose", "scale"])#copy(optTrans)
    optTrans_new["pose"] = [0,0]
    optTrans_new["scale"] = 1
    initPose = np.array([optTrans_new["pose"][0],
                         # 0,
                         optTrans_new["pose"][1],
                         1#optTrans_new["scale"]
                        ])
    bounds = np.array([
                      (-20, 20),
                      (-20, 20),
                      # (optTrans_new["pose"][0]-20, optTrans_new["pose"][0]+20),#edgePoints_mm[:,0].max()),
                      # #(-np.inf, np.inf),
                      # (optTrans_new["pose"][1]-10, optTrans_new["pose"][1]+10),
                      (0.4, 2.0),#(optTrans_new["scale"]*0.9, optTrans_new["scale"]*1.1),
                      (-3,3)])#(-3, 3)])

    assam.optimiseStage = "both" # tell class to optimise shape and pose
    assam.optIterSuc, assam.optIter = 0, 0
    assam.scale = optTrans_new["scale"]
    assert len(assam.projLM_ID) != 0, 'no projected landmarks'
    optAll = assam.optimiseAirwayPoseAndShape(assam.objFuncAirway, 
                                               initPose, 
                                               bounds, 
                                               epochs=numEpochs, 
                                               threads=1)
    print("\n\n\n\n\n\t\t\tTime taken is {0} ({1} mins)".format(time()-startTime,
                                                        round((time()-startTime)/60.),3))
    outShape = assam.morphAirway(inputCoords['ALL'],
                                inputCoords['ALL'].mean(axis=0),
                                optAll["b"],
                                assam.model_s['ALL'][:len(optAll["b"])]
                                )
    outShape = assam.centerThenScale(outShape, optAll['scale'], outShape.mean(axis=0))


    out_file = '{}_{}.{}'
    out_surf_file = 'surfaces/'+out_file
    out_lm_file = 'outputLandmarks/'+out_file
    np.savetxt(out_lm_file.format(tag, 'ALL', 'csv'), outShape,
                header='x, y, z', delimiter=',')
    '''
    out_airway_file = out_surf_file.format(tID, 'Airway', 'stl')
    morph_airway = MorphAirwayTemplateMesh(lm_template[lmOrder['Airway']], 
                                            outShape[lmOrder['Airway']], 
                                            mesh_template,
                                            quiet=True)
    morph_airway.mesh_target.write(out_airway_file)
    for lobe in lobes:
      print('morphing', lobe)
      # lm_template_lobes = np.loadtxt(template_lm_file_lobes.format(key), delimiter=",")
      template_lobe_mesh = v.load(template_mesh_file_lobes.format(lNums[lobe]))
      # template_lobe_mesh.decimate(fraction=0.25)
      morph_lobe = MorphLobarTemplateMesh(lm_template[lmOrder[lobe]], 
                                          outShape[lmOrder[lobe]], 
                                          template_lobe_mesh,
                                          quiet=True)
      out_lobe_file = out_surf_file.format(tID, lobe, 'stl')
      morph_lobe.mesh_target.write(out_lobe_file)
    '''
    plt.close()
    plt.imshow(img, cmap='gray', extent=extent)
    plt.scatter(outShape[:,0], outShape[:,2],s=2,c='black')
    plt.savefig('images/reconstruction/{}.png'.format(tag), dpi=200)

    # shape parameters for ground truth
    b_gt = getShapeParameters(inputCoords['ALL'], tLM, 
                               assam.model_s['ALL'], assam.std)
    shape_parameter_diff = (b_gt-optAll['b'])
    print('parameter difference is')
    print(shape_parameter_diff)

    distX = utils.euclideanDist(outShape[:,[0]], lmProj_test[0][:,[0]])
    dist2D = utils.euclideanDist(outShape[:,[0,2]], lmProj_test[0][:,[0,2]])

    # ax[0].imshow(img, cmap="gray", extent=extent) 
    # ax[0].scatter(edgePoints[:,0], edgePoints[:,1],s=2) 
    # ax[1].scatter(meanArr[:,0], meanArr[:,2],s=1, c="black") 
    # ax[1].scatter(edgePoints[:,0], edgePoints[:,1],s=2, c="blue") 

    '''
ourShape = outShape - outShape[lmOrder['SKELETON'][1]]
projTest = copy(lmProj_test[0])
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
    '''
    '''

airwayProj_pts = lmOrder['Airway'][np.isin(lmOrder['Airway'], assam.projLM_ID['Airway'])]
ourShape = (outShape - outShape[lmOrder['SKELETON'][1]])[airwayProj_pts]
projTest = copy(lmProj_test[0])[airwayProj_pts]
plt.close()
plt.scatter(ourShape[:,0], ourShape[:,2], s=2, c='blue')
plt.scatter(projTest[:,0], projTest[:,2], s=2, c='black')
plt.show()
    '''
