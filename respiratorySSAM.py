'''
 User args:
  --inp
    set landmark directory
  --drrs
    set radiograph directory (drr = digitally reconstructed radiograph)
  --debug
    use debug flags
  --getModes
    write main modes of variation to surface

  Develops SSAM  based on landmarks determined by GAMEs algorithm
  and DRRs extracted from the same CT dataset.

  @author: Josh Williams
'''
# Note to future developers:
#   SSAM has been developed to be generic, i.e. that it can easily be
#   implemented for appearance modelling of other organs, or images in general.
#   When adapting to other organs, images etc, some changes may be required
#   such as user arguments and data structure (file names etc).


import numpy as np
import matplotlib.pyplot as plt
import random
import vtk

from vedo import *
import argparse
from copy import copy
from os import path
from sys import exit, argv
from glob import glob
from sklearn.model_selection import train_test_split, \
                  LeaveOneOut,\
                  KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from skimage import io
from skimage.color import rgb2gray

from scipy.spatial.distance import cdist, pdist

#plot graphs for compactness, specificity etc
from ssmPlot import *
# from userUtils import cm2inch, doPCA, trainTestSplit
import userUtils as utils

from respiratorySAM import RespiratorySAM
from respiratorySSM import RespiratorySSM

class RespiratorySSAM:

  def __init__(self, lm, lm_ct, imgs, imgsOrigin, imgsSpacing,
                testSplit="trainTestSplit", train_size=0.9):
    # check all input datasets have same number of samples
    assert lm.shape[0]==imgs.shape[0]==imgsOrigin.shape[0]==imgsSpacing.shape[0], \
      'non-matching dataset size'
    #-import functions
    self.doPCA = utils.doPCA
    self.trainTestSplit = utils.trainTestSplit

    #-shape modelling classes
    self.sam = RespiratorySAM(lm_ct, imgs, imgsOrigin, imgsSpacing)
    self.ssm = RespiratorySSM(lm)

    #-initialise input variables
    self.lm = lm # centered landmarks from GAMEs
    self.lm_ct = lm_ct # landmarks transformed to same coordinate frame as CT
    self.train_size = train_size

    #-initialise appearance model data
    self.imgs = imgs
    self.imgsN = self.sam.imgsN
    self.imgCoords = self.sam.imgCoords
    self.density = self.sam.density
    
    #-initialise shape model data
    self.x_vec = self.lm.reshape(self.lm.shape[0],
                                  self.lm.shape[1]* 
                                  self.lm.shape[2])
    # self.x_vec_scale = StandardScaler().fit_transform(self.x_vec)
    self.x_vec_scale = self.x_vec - self.x_vec.mean(axis=1)[:,np.newaxis]
    self.x_vec_scale /= self.x_vec_scale.std(axis=1)[:,np.newaxis]
    # self.x_vec_scale = self.ssm.x_vec_scale

    self.x_train = self.ssm.x_vec_scale
    self.g_train = self.density
    # self.x_train, self.x_test = self.ssm.x_train, self.ssm.x_test
    # self.g_train, self.g_test = self.sam.g_train, self.sam.g_test

    #-initialise models -- commented as not needed -> avoids additional training
    # self.pca_s = self.ssm.pca #-train SSM and get PCs
    # self.pca_g = self.sam.pca
    # self.phi_s = self.pca_s.components_ # shape components
    # self.phi_g = self.pca_g.components_ # appearance components

    # #-initialise empty vector. Can only be same size as training set
    self.b_s = np.zeros(self.x_train.shape[0])
    self.b_g = np.zeros(self.g_train.shape[0])
    self.b_sg = np.zeros(self.g_train.shape[0])

    
    self.xg = np.dstack((self.x_vec_scale.reshape(self.lm.shape), 
                         self.density))
    self.xg_vec = self.xg.reshape(self.xg.shape[0],
                                  self.xg.shape[1]* 
                                  self.xg.shape[2])
    
    self.pca_sg = self.buildSSAM_singlePCA(self.xg_vec)
    self.phi_sg = self.pca_sg.components_
    self.variance = self.pca_sg.explained_variance_
    self.std = np.sqrt(self.variance)
    self.xg_train = self.xg_train.reshape(-1,self.xg.shape[1],self.xg.shape[2])

  def testLandmarkAndImageBounds(self, lm, imgCoords):
    minCheck = imgCoords.min(axis=1)[:,0] < lm.min(axis=1)[:,0]
    maxCheck = imgCoords.max(axis=1)[:,0] > lm.max(axis=1)[:,0]
    assert maxCheck.all() and minCheck.all(), 'shape is outside image bounds'

  def buildSSAM_singlePCA(self, xg_vec):
    '''
      Input: 2D array with shape (nSamples, 4 x nLandmarks)
      Return: PCA object
    '''
    printc("Building SSAM",c="white")
    # self.xg_train = self.trainTestSplit(xg_vec, self.train_size)[0]
    self.xg_train = self.xg_vec
    pca = self.doPCA(self.xg_train, 0.95)[0]
    return pca

def getInputs():
  parser = argparse.ArgumentParser(description='SSAM')
  parser.add_argument('-i','--inp', 
                      default='allLandmarks/', 
                      type=str, 
                      help='input files (landmarks)'
                      )
  parser.add_argument('--drrs',  
                      default='../xRaySegmentation/DRRs_enhanceAirway/luna16_cannyOutline/', 
                      type=str, 
                      help='input files (drr)'
                      )
  parser.add_argument('--getModes', 
                      default=False, 
                      type=str, 
                      help='write main modes of variation'
                      )
  parser.add_argument('--debug', 
                      default=False, 
                      type=bool, 
                      help='debug mode -- shows print checks and blocks '+
                          'plotting outputs'
                      )

  args = parser.parse_args()
  landmarkDir = args.inp
  drrDir = args.drrs

  assert path.isdir(landmarkDir) and path.isdir(drrDir), 'data directories do not exist'

  debugMode = args.debug 
  getModes = args.getModes


  return landmarkDir, drrDir, debugMode, getModes

if __name__ == "__main__":
  print(__doc__)

  landmarkDir, drrDir, debug, getModes = getInputs()

  trainSplit = 0.9

  printc("shape and appearance modelling", c="green")
  #-Get directories for DRR and landmark data
  originDirs = glob( drrDir + "/origins/origins/drr*.md")#.sort()
  spacingDirs = glob( drrDir + "/*/drr*.md")#.sort()
  imDirs = glob(drrDir + "/*/drr*.png")#.sort()
  originDirs.sort()
  spacingDirs.sort()
  imDirs.sort()
  patientIDs = [i.split("/")[-1].replace(".png", "")[-4:] for i in imDirs]
  landmarkDirs = glob( landmarkDir+"/*andmarks*.csv" )
  # landmarkDirs = sorted(landmarkDirs, 
  #                       key=lambda x: int(x.replace(".csv","")[-4:]))
  landmarkDirs.sort()
  lmIDs = [i.split("/")[-1].split("Landmarks")[1][:4] for i in landmarkDirs]
  landmarkDirsOrig = glob('landmarks/manual-jw/landmarks*.csv')
  landmarkDirsOrig.sort()
  #-used to align drrs and landmarks
  # transDirs = glob( drrDir+"/transforms/transforms/allLandmarks/"
  #                             +"transformParams_case*"
  #                             +"_m_*.dat")
  # transDirs.sort()

  if len(imDirs) == 0 \
  or len(originDirs) == 0 \
  or len(landmarkDirs) == 0 \
  or len(spacingDirs) == 0 \
  or len(landmarkDirsOrig) == 0: 
    print("ERROR: The directories you have declared are empty.",
          "\nPlease check your input arguments.")
    print(drrDir, len(imDirs))
    print(landmarkDir, len(landmarkDirs))
    print('landmarks/manual-jw/', len(landmarkDirsOrig))
    exit()

  delInd = []
  for i, imD in enumerate(imDirs):
    currID = imD.split('.')[-2][-4:]
    if currID not in lmIDs:
      delInd.append(i)

  for dId in delInd[::-1]:
    originDirs.pop(dId)
    spacingDirs.pop(dId)
    imDirs.pop(dId)

  # landmarkTrans = np.vstack([np.loadtxt(t, skiprows=1, max_rows=1) 
  #                           for t in transDirs])
  #-read data
  origin = np.vstack([np.loadtxt(o, skiprows=1)] for o in originDirs)
  spacing = np.vstack([np.loadtxt(o, skiprows=1)] for o in spacingDirs)
  #-load x-rays into a stacked array, 
  #-switch so shape is (num patients, x pixel, y pixel)
  drrArr = np.rollaxis(
                      np.dstack([utils.loadXR(o) for o in imDirs]),
                      2, 0)
  nodalCoords = np.array([np.loadtxt(l, delimiter=",",skiprows=1) 
                            for l in landmarkDirs])
  nodalCoordsOrig = np.array([np.loadtxt(l, delimiter=",",skiprows=1,usecols=[1,2,3]) 
                            for l in landmarkDirsOrig])
  carinaArr = nodalCoordsOrig[:,1]
  
  #-offset centered coordinates to same reference frame as CT data
  lmProj = nodalCoords + carinaArr[:,np.newaxis]#landmarkTrans[:, np.newaxis]
  #-create appearance model instance and load data
  ssam = RespiratorySSAM(nodalCoords, lmProj, drrArr, origin, spacing,
            "trainTestSplit", train_size=trainSplit)
  drrPos = ssam.imgCoords
  drrArrNorm = ssam.imgsN
  density = ssam.g_train
  phi_sg = ssam.phi_sg
  if debug:
    ssam.testLandmarkAndImageBounds(ssam.lm[:,:,[0,2]], ssam.imgCoords)

  if getModes:
    printc("plotting modes of shape + appearance variation...", c="blue")
    modes = [0,1,2,3]
    stds = [3, -3] 
    meanArr_sg = ssam.xg_train.mean(axis=0)
    plt.close()
    fig, ax = plt.subplots(nrows=len(stds),ncols=len(modes))#,
                           #figsize=cm2inch(17,10))
    for i, mode in enumerate(modes):
      ax[0][i].set_title("mode "+str(mode+1), fontsize=11)
      for j, std in enumerate(stds):
        if i==0:
          if std!=0:
            title = "SD = "+str(std)
          else:
            title = "mean"
          ax[j][0].set_ylabel(title, rotation=90, fontsize=11)
        b = ssam.b_sg
        b[mode] = std 
        # var = np.dot(ssam.phi_sg.T, b).reshape(meanArr_sg.shape)
        var = (ssam.phi_sg.T * b * ssam.std)[:,mode].reshape(meanArr_sg.shape)
        shapeVar = meanArr_sg + var

        a = ax[j][i].scatter(shapeVar[:,0], shapeVar[:,2], 
                              c=var[:,-1], 
                              cmap='seismic',#'RdBu_r',
                              vmin=-3, vmax=3,
                              s=1)

        ax[j][i].xaxis.set_ticklabels([])
        ax[j][i].yaxis.set_ticklabels([])
        ax[j][i].spines['top'].set_visible(False)
        ax[j][i].spines['right'].set_visible(False)
        ax[j][i].spines['bottom'].set_visible(False)
        ax[j][i].spines['left'].set_visible(False)
        ax[j][i].get_xaxis().set_ticks([])
        ax[j][i].get_yaxis().set_ticks([])
    cb = fig.colorbar(a, ax=ax.ravel().tolist())
    cb.set_label("normalised density change", fontsize=11)
    fig.text(0.5, 0.95, "shape & appearance components",
            horizontalalignment='center', fontsize=12)
    fig.savefig("images/SSAM/SSAM-modes-diff.png", 
                 pad_inches=0, format="png", dpi=300)

    #-plot modes of variation with colour map of density at each landmark
    plt.close()
    fig, ax = plt.subplots(nrows=len(stds),ncols=len(modes))#,
    for i, mode in enumerate(modes):
      ax[0][i].set_title("mode "+str(mode+1), fontsize=11)
      for j, std in enumerate(stds):
        if i==0:
          if std!=0:
            title = "SD = "+str(std)
          else:
            title = "mean"
          ax[j][0].set_ylabel(title, rotation=90, fontsize=11)
        #-vary shape by prescribed deviation
        b = ssam.b_sg
        b[mode] = std 
        var = (ssam.phi_sg.T * b * ssam.std)[:,mode].reshape(meanArr_sg.shape)
        shapeVar = meanArr_sg + var
        #-change density from 0 mean, 1 std to normal scale
        modeIm = (var[:,-1]
                  *ssam.sam.density_base.std(axis=0)
                  )+ssam.sam.density_base.mean(axis=0)
        a = ax[j][i].scatter(shapeVar[:,0], shapeVar[:,2], 
                            c=modeIm, 
                            cmap='gray',
                            vmin=0, vmax=1, 
                            s=1)

        ax[j][i].xaxis.set_ticklabels([])
        ax[j][i].yaxis.set_ticklabels([])
        ax[j][i].spines['top'].set_visible(False)
        ax[j][i].spines['right'].set_visible(False)
        ax[j][i].spines['bottom'].set_visible(False)
        ax[j][i].spines['left'].set_visible(False)
        ax[j][i].get_xaxis().set_ticks([])
        ax[j][i].get_yaxis().set_ticks([])
    cb = fig.colorbar(a, ax=ax.ravel().tolist())
    cb.set_label("density", fontsize=11)
    fig.text(0.5, 0.95, "shape & appearance components",
            horizontalalignment='center', fontsize=12)
    fig.savefig("images/SSAM//SSAM-modes.png", 
                 pad_inches=0, format="png", dpi=300)
