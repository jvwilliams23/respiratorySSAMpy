'''
    run script for reconstructing airways from an X-ray
'''

from concurrent import futures
import userUtils as utils
import numpy as np
import matplotlib.pyplot as plt
import random
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
                      default='landmarks/manual-jw-diameterFromSurface/', 
                      type=str, 
                      help='input files (landmarks)'
                      )
  parser.add_argument('--case', '-c',
                      default=False, 
                      type=str,#, required=True,
                      help='training data case'
                      )
  parser.add_argument('--out', '-o',
                      default='reconstructedAirway', 
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
  parser.add_argument('--debug', 
                      default=False, 
                      type=bool, 
                      help='debug mode -- shows print checks and blocks'+
                            'plotting outputs'
                      )
  parser.add_argument('--epochs', '-e',
                      default=5000, 
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
  inputDir = args.inp
  tag = args.out
  case = args.case
  var = args.var
  drrDir = args.drrs
  debugMode = args.debug 
  shapeKey = args.shapes.split()
  surfDir = args.meshdir
  numEpochs = args.epochs
  xray = args.xray
  c_edge = args.c_edge
  c_dense = args.c_dense
  c_prior = args.c_prior
  imgSpacing = args.imgSpacing

  return inputDir, case, tag, var, drrDir, \
          debugMode, shapeKey, surfDir, numEpochs, \
          xray, c_edge, c_dense, c_prior, imgSpacing

if __name__=='__main__':
  date_today = str(date.today())
  print(__doc__)
  startTime = time()


  landmarkDir, case, tag, describedVariance, drrDir, debug, \
          shapes, surfDir, numEpochs, xrayEdgeFile, \
          c_edge, c_dense, c_prior, imgSpaceCoeff = getInputs()
  img=None
  spacing_xr=None
  template_lmFileOrig = 'landmarks/manual-jw/landmarks3948.csv'
  # template_lmFile = 'landmarks/manual-jw-diameterFromSurface/landmarks3948_diameterFromSurf.csv'
  template_lmFile = 'allLandmarks/allLandmarks3948.csv'
  template_meshFile = 'segmentations/template3948/newtemplate3948_mm.stl'

  print("\tReading data")
  #-read DRR data
  originDirs = glob( drrDir + "/origins/origins/drr*.md")#.sort()
  spacingDirs = glob( drrDir + "/*/drr*.md")#.sort()
  imDirs = glob(drrDir + "/*/drr*.png")#.sort()
  originDirs.sort()
  spacingDirs.sort()
  imDirs.sort()

  patientIDs = [i.split("/")[-1].replace(".png", "")[-4:] for i in imDirs]
  landmarkDirs = glob( landmarkDir+"/*andmarks*.csv" )
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

  # landmarkFiles = glob(landmarkDir+"landmark*"+shape+"*")
  # landmarkFiles = sorted(landmarkFiles, 
  #                       key=lambda x: int(x.split("case")[1][:4]))
  landmarks = np.array([np.loadtxt(l, delimiter=",",skiprows=1) 
                            for l in landmarkDirs])
  nodalCoordsOrig = np.array([np.loadtxt(l, delimiter=",",skiprows=1,usecols=[1,2,3]) 
                            for l in landmarkDirsOrig])

  #-read appearance modelling data
  origin = np.vstack([np.loadtxt(o, skiprows=1)] for o in originDirs)
  spacing = np.vstack([np.loadtxt(o, skiprows=1)*imgSpaceCoeff] 
                              for o in spacingDirs)
  # crop last two rows of pixels off XR so white pixels don't interfere with normalising
  drrArr = np.rollaxis(
                      np.dstack([utils.loadXR(o)[:-2,:-2] for o in imDirs]),
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
  for t in testSet[::-1]:
    #-store test data in different list
    testID.append(patientIDs[t])
    testOrigin.append(origin[t])
    testSpacing.append(spacing[t])
    testIm.append(drrArr[t])
    testLM.append(landmarks[t])
    #-remove test data from train data
    patientIDs.pop(t)
    origin = np.delete(origin,t,axis=0)
    spacing = np.delete(spacing,t,axis=0)
    drrArr = np.delete(drrArr,t,axis=0)
    landmarks = np.delete(landmarks,t,axis=0)
    lmProj = np.delete(lmProj,t,axis=0)

  mean_shape_file = surfDir+'meanAirway.stl'
  meanArr = landmarks.mean(axis=0)

  # template_lm_file_lobes = landmarkDir+"/landmarks0{}-case8684.csv"
  template_mesh_file_lobes = "/home/josh/3DSlicer/project/luna16Rescaled/case3948/3948_mm_{}.stl"
  lm_template = np.loadtxt(template_lmFile, skiprows=1, delimiter=",",
                              usecols=[0,1,2])
  carinaTemplate = np.loadtxt(template_lmFileOrig, skiprows=1, delimiter=',',
                              usecols=[1,2,3])[1]*-1




  #-load mesh data and create silhouette


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
  lobes = ['RUL', 'RML', 'RLL', 'LUL', 'LLL']
  # numbering for each lobe in file
  lNums = {"RUL": "4",
               "RML": "5",
               "RLL": "6",
               "LUL": "7",
               "LLL": "8"
               }
  lmOrder = dict.fromkeys(shapes)
  modelDict = dict.fromkeys(shapes)
  inputCoords = dict.fromkeys(shapes)
  inputCoords['ALL'] = meanArr
  modelDict['ALL'] = model
  for shape in shapes:
    lmOrder[shape] = np.loadtxt(landmarkDir+'landmarkIndex{}.txt'.format(shape),
                                dtype=int)
    inputCoords[shape] = meanArr[lmOrder[shape]]
    modelDict[shape] = model.reshape(len(landmarks), -1, 4)[:,lmOrder[shape]]

  mesh_template = v.load(template_meshFile)
  mesh_template = mesh_template.pos(carinaTemplate)
  mean_mesh = dict.fromkeys(shapes)
  faces = dict.fromkeys(shapes) # faces of mean surface for each shape
  surfCoords_mmOrig = dict.fromkeys(shapes) # surface nodes for each shape
  surfToLMorder = dict.fromkeys(shapes) # mapping between surface nodes and LMs
  newMean = False
  # create mesh for population average from a morphing algorithm
  if glob(mean_shape_file) == 0 or newMean:
    morph_airway = MorphAirwayTemplateMesh(lm_template[lmOrder['Airway']], 
                                            meanArr[lmOrder['Airway']], 
                                            mesh_template)
    morph_airway.mesh_target.write(mean_shape_file)
    for lobe in lobes:
      # lm_template_lobes = np.loadtxt(template_lm_file_lobes.format(key), delimiter=",")
      template_lobe_mesh = v.load(template_mesh_file_lobes.format(lNums[lobe]))
      morph_lobe = MorphLobarTemplateMesh(lm_template[lmOrder[lobe]], 
                                          meanArr[lmOrder[lobe]], 
                                          template_lobe_mesh)
      mean_lobe_file_out = surfDir+'mean{}.stl'.format(lobe)
      morph_lobe.mesh_target.write(mean_lobe_file_out)

  for key in shapes:
    mean_shape_file = surfDir+'mean{}.stl'.format(key)
    mean_mesh[key] = v.load(mean_shape_file).computeNormals()

  #-reorder unstructured stl file to be coherent w/ model and landmarks
  #-extract mesh data (coords, normals and faces)
  for key in shapes:
    print('loading {} mesh'.format(key))
    mesh = mean_mesh[key]
    surfCoords = mesh.points()
    meanNorms_face = mesh.normals(cells=True)
    faces[key] = np.array(mesh.faces())

    #-offset to ensure lobes are stacked 
    surfCoords_mm = surfCoords + carinaArr.mean(axis=0)
    surfCoords_mmOrig[key] = copy(surfCoords_mm)
    surfToLMorder = []
    for point in meanArr[lmOrder['Airway']]:
      surfToLMorder.append( np.argmin( utils.euclideanDist(surfCoords, 
                                                                    point) ) )

  tagBase = copy(tag)
  for t, (tID, tImg, tOrig, tSpace) \
    in enumerate(zip(testID, testIm, testOrigin, testSpacing)):

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

    # inputCoords = meanArr.copy()
    ##########################################################################
    #-declare posterior shape model class
    assam = RespiratoryReconstructSSAM(shape=inputCoords,
                                      xRay=edgePoints_mm,
                                      lmOrder=lmOrder,
                                      normals=None, 
                                      transform=carinaArr[t],
                                      img=img,
                                      # imgSpacing=spacing_xr,
                                      imgCoords=imgCoords,
                                      density=density,
                                      model=modelDict,
                                      modeNum=numModes,
                                      c_edge=c_edge,
                                      c_prior=c_prior,
                                      c_dense=c_dense)
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
    t1 = time()
    assam.projLM, assam.projLM_ID = assam.getProjectionLandmarks(faces, 
                                                                meanNorms_face, 
                                                                surfCoords_mmOrig)

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
    lobeBackup = copy(assam)

    # assam.projLM, assam.projLM_ID = assam.deleteShadowedEdges(surfCoords_mm, 
    #                                                       assam.projLM, 
    #                                                       assam.projLM_ID,
    #                                                       )

    #-reorder projected surface points to same order as landmarks
    delInd = []
    print('reordering projected landmarks')
    for p, point in enumerate(assam.projLM_ID):
      if np.isin(surfToLMorder, point).sum() > 0: #np.isin(point, surfToLMorder):
        mappedLM = np.argwhere(np.isin(surfToLMorder, point))
        assam.projLM_ID[p] = mappedLM[0][0]
      else:
        delInd.append(p)
    print('finished reordering projected landmarks')

    #-delete projected surfPoints which were not included in mapping to LM space
    assam.projLM_ID = np.delete(assam.projLM_ID, delInd)
    assam.projLM = np.delete(assam.projLM, delInd)
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
    plt.close()
    plt.imshow(img, cmap='gray', extent=extent)
    plt.scatter(outShape[:,0], outShape[:,2],s=2,c='black')
    plt.savefig('images/reconstruction/test{}.png'.format(t), dpi=200)
    # plt.show()

    outAirway = outShape[np.array(lmOrder['Airway'])]
    morph = MorphAirwayTemplateMesh(lm_template, outAirway, mesh_template)
    morph.mesh_target.write('out3948.stl')

    # ax[0].imshow(img,cmap="gray", extent=extent) 
    # ax[0].scatter(edgePoints[:,0], edgePoints[:,1],s=2) 
    # ax[1].scatter(meanArr[:,0], meanArr[:,2],s=1, c="black") 
    # ax[1].scatter(edgePoints[:,0], edgePoints[:,1],s=2, c="blue") 
