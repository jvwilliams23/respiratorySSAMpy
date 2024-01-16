'''
  Develops SSM of lobes and airways

  @author: Josh Williams

'''

import numpy as np
import matplotlib.pyplot as plt
import nevergrad as ng
import vedo as v
import networkx as nx 

from scipy.spatial.distance import cdist

import argparse
from distutils.util import strtobool
from re import findall
from copy import copy
from sys import exit
from glob import glob
from datetime import date

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

#plot graphs for compactness, specificity etc
# from ssmPlot import *
import userUtils as utils
from morphAirwayTemplateMesh import MorphAirwayTemplateMesh

class RespiratorySSM:

  def __init__(self, lm, train_size=0.9, quiet=False, normalise=True):

    #-import functions
    self.doPCA = utils.doPCA
    self.trainTestSplit = utils.trainTestSplit

    self.lm = lm
    # self.d = diameter
    self.x_scale = self.lm.copy() #
    # self.x_scale = self.lm - self.lm.mean(axis=1)[:,np.newaxis,:]
    # self.x_scale /= self.x_scale.std(axis=1)[:,np.newaxis,:] # jw 26/04/21
    # self.x_scale /= self.x_scale.std(axis=0)

    nodiameter = True
    self.x_vec_scale = self.x_scale.reshape(len(lm), -1)
    self.shape = 3
    if normalise:
      self.x_vec = self.lm.reshape(lm.shape[0], lm.shape[1]*lm.shape[2])
      self.x_vec_scale = self.x_vec - self.x_vec.mean(axis=1)[:,np.newaxis]
      self.x_vec_scale = self.x_vec_scale / self.x_vec_scale.std(axis=1)[:,np.newaxis]

    # if type(self.d)==None:
    #   self.x_n3_scale = self.x_vec_scale.reshape(lm.shape[0], -1, 4)
    # else:
    self.x_n3_scale = self.x_vec_scale.reshape(lm.shape[0], -1, self.shape)


    ''' TODO: instead of splitting here, create an input arg as np array with
    split indexes i.e. arr1 has 95% of data with indexes 0-max, arr2 has rest
    if no arg provided, then can split here. This gives the option to the user
    to use split on other data i.e. caseIDs without explicitly calling caseIDs
    to the class (it is not general enough to be needed in all class instances)
    '''
    self.x_train, self.x_test = self.trainTestSplit(self.x_vec_scale,
                                                    train_size,
                                                    quiet=quiet)

    self.x_scale_tr = self.x_train
    self.x_scale_te = self.x_test

    if __name__=="__main__":
      self.pca, self.k = self.doPCA(self.x_train, 0.95, quiet=quiet) #-train the model
      self.phi = self.pca.components_ #-get principal components
      self.variance = self.pca.explained_variance_
      self.std = np.sqrt(self.variance)
      self.phi_s = self.filterModelShapeOnly(self.phi)
      if not nodiameter:
        self.phi_d = self.filterModelDiameterOnly(self.phi)

    #-initialise empty vector. Can only be same size as training set
    self.b = np.zeros(self.x_train.shape[0])


  def filterModelShapeOnly(self, model):
    '''
      Return model without density in columns.
      Input 2D array, shape = ( nFeature, 4n )
      Return 2D array, shape = ( nFeature, 3n )
      where n = num landmarks
    '''
    #no appearance params
    model_noApp = model.reshape(model.shape[0],-1, self.shape) 
    return model_noApp[:,:,[0,1,2]].reshape(model.shape[0],-1) # -reshape to 2D array

  def filterModelDiameterOnly(self, model):
    '''
      Return model without shape in columns.
      Input 2D array, shape = ( nFeature, 4n )
      Return 2D array, shape = ( nFeature, n )
      where n = num landmarks
    '''
    #no shape params
    model_noSh = model.reshape(model.shape[0],-1, self.shape) 
    return model_noSh[:,:,-1].reshape(model.shape[0],-1) # -reshape to 2D array


  def solve_for_b(self, shape_base, shape_target, 
                    b, objective,
                    epochs=500, threads=1):
    '''
        Minimises objective function using nevergrad optimiser
    '''
    instrum = ng.p.Instrumentation(
                                    b=ng.p.Array(
                                                init=np.zeros(b.size)
                                                ).set_bounds(-3,3)
                                    )
    optimizer = ng.optimizers.NGO(#CMA(#NGO(
                                 parametrization=instrum, 
                                 budget=epochs, 
                                 num_workers=threads)

    if threads > 1:
      with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
          recommendation = optimizer.minimize(objective, \
                                              executor=executor, \
                                              batch_mode=True)
    else:
      # recommendation = optimizer.minimize(objective)
      lossLog = []
      recommendation = optimizer.provide_recommendation()
      for _ in range(optimizer.budget):
          # print("testing")
          x = optimizer.ask()
          btmp = x.value[1]["b"]
          # print("X check", )
          # loss = objective(*x.args, **x.kwargs)
          loss = objective(btmp*self.std,shape_base,shape_target)
          optimizer.tell(x, loss)
          lossLog.append(loss)

    recommendation = optimizer.provide_recommendation() #-update recommendation

    return  recommendation.value[1]["b"]

  def computeMean(self, lm):
    '''
      Args:
        landmarkFiles

      Returns:
        mean value of each landmark index to create the mean shape
    ''' 
    avg = np.mean(lm, axis=0)
    return avg

def getInputs():
  parser = argparse.ArgumentParser(description='SSM args')
  parser.add_argument('-i','--inp', 
                      default='allLandmarks_noFissures/', 
                      type=str, 
                      help='input files (landmarkFiles)'
                      )
  parser.add_argument('--getMetrics', 
                      default=False, 
                      type=bool, 
                      help='process SSM metrics (compactness etc)'
                      )
  parser.add_argument('--getModes', 
                      default=False, 
                      type=str, 
                      help='write main modes of variation to surface'
                      )
  parser.add_argument('--writeMean', 
                      default=False, 
                      type=bool, 
                      help='write mean point cloud from all landmarkFiles'
                      )
  parser.add_argument('--normalise', 
                      default=str(False), 
                      type=strtobool, 
                      help='normalise landmarks before training model'
                      )
  parser.add_argument('--debug', 
                      default=False, 
                      type=str, 
                      help='debug mode -- shows print checks and blocks'+
                          'plotting outputs'
                      )

  return parser.parse_args()

labels = ['topTrachea','endTrachea', 'endRMB', 'endLMB', 
          'triEndBronInt', 'triRUL', 'triLUL', 'llb6']

if __name__ == "__main__":
  print(__doc__)

  args = getInputs()

  landmarkDir = args.inp
  debug = args.debug 
  getMetrics = args.getMetrics
  getModes = args.getModes
  writeMean = args.writeMean
  normalise = args.normalise
  # landmarkDir, debug, getMetrics, getModes, writeMean, normalise = getInputs()
  num_intermediatePts = 15 #25
  # normalise = False

  trainSplit = 0.99

  landmarkFiles = glob( landmarkDir+"/allLandmarks*.csv")
  if len(landmarkFiles) == 0:
    print("ERROR: The directories you have declared are empty.",
          "\nPlease check your input arguments.")
    exit()
  landmarkFilesOrig = glob('landmarks/manual-jw/landmarks*.csv')
  landmarkFiles.sort()
  landmarkFilesOrig.sort()
  caseIDs = [findall('\d+', f)[0] for f in landmarkFiles]

  # load graphs - bgraph is branches only, vgraph is all voxels
  bgFile = 'skelGraphs/nxGraph{}BranchesOnly.pickle'
  vgFile = 'skelGraphs/nxGraph{}.pickle'
  lgFile = 'landmarks/manual-jw-diameterFromSurface/nxGraph{}landmarks.pickle'
  meshFile = 'segmentations/airwaysForRadiologistWSurf/{}/*.stl'
  bgraphList = [nx.read_gpickle(bgFile.format(cID)) for cID in caseIDs]
  lgraphList = [nx.read_gpickle(lgFile.format(cID)) for cID in caseIDs]
  vgraphList = [nx.read_gpickle(vgFile.format(cID)) for cID in caseIDs]

  # load landmarks
  nodalCoordsOrig = np.array([np.loadtxt(l, skiprows=1, delimiter=",",
                                      usecols=[1,2,3]) 
                            for l in landmarkFilesOrig])
  nodalCoords = np.array([np.loadtxt(l, skiprows=1, delimiter=",",
                                      usecols=[0,1,2]) 
                            for l in landmarkFiles])
  template_lmFileOrig = 'landmarks/manual-jw/landmarks3948.csv'
  template_lmFile = 'landmarks/manual-jw-diameterFromSurface/landmarks3948_diameterFromSurf.csv'
  template_meshFile = 'segmentations/template3948/newtemplate3948_mm.stl'

  posList = []
  for i, lgraph in enumerate(lgraphList):
    pos = []
    for node in lgraph.nodes:
      pos.append(lgraph.nodes[node]['pos'])
    pos = np.vstack(pos)
    posList.append(pos)

  nodalCoords = utils.procrustesAlignVedo(nodalCoords)

  diameter = np.ones(nodalCoords.shape[:-1]) # dummy placeholder
  
  #initialise shape model
  diameter = np.ones(nodalCoords.shape[:-1])
  ssm = RespiratorySSM(nodalCoords, train_size=trainSplit, normalise=normalise)
  xBar = ssm.computeMean(ssm.x_scale_tr)
  xBar_n3 = xBar.reshape(-1,ssm.shape)
  dBar = xBar_n3[:,-1]
  xBar_n3 = xBar_n3[:,[0,1,2]]
