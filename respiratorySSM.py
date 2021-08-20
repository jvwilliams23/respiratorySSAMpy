'''
  Develops SSM of lobes and airways

  @author: Josh Williams

'''

import numpy as np
import matplotlib.pyplot as plt
import vtk
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
from sklearn.preprocessing import StandardScaler
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

  def objFunc(self, b, shape_base, shape_target):
    '''
      function to determine shape parameters (b) by minimising distance between
      modelled shape (shape_base, vector) and target (shape_target, vector)
    '''
    shapeIn = self.getx_allModes(shape_base, self.phi, b) 

    d = utils.euclideanDist(shapeIn, shape_target)
    dNorm = np.sum(np.exp(-d/5))/d.shape[0]
    return 1-dNorm

  def testReconstruction(self, trainData, mean_tr, phi_tr, k, pca):
    if k < 1:
      print("LOW k found")
      k = np.where(np.cumsum(pca.explained_variance_ratio_)>k)[0][0]
      print("num components is ", k)

    r_k = 0 #initialise reconstruction error

    for n in range(trainData.shape[0]):
      s_i = trainData[n].reshape(-1,3)
      # normalise = True
      if normalise:
        max_dist = 100.
        max_dist = abs(s_i.max(axis=0) - s_i.min(axis=0)).max()
        # max_dist = cdist(s_i, s_i).max()
      b = np.dot((s_i.reshape(-1) - mean_tr), phi_tr.T)/self.std #-actually b * std, not b
      s_i_k = self.getx_allModes(mean_tr, phi_tr[:k], b[:k]*self.std[:k])

      # r_k_i = np.sum(abs(s_i_k - s_i))/len(s_i) # reconstruction error for instance i
      r_k_i = np.sum(utils.euclideanDist(s_i_k, s_i))/len(s_i) # reconstruction error for instance i

      if debug:
        print("r_k")
        print(r_k.shape)
        print("SHAPE CHECK")
        print(s_i.shape, s_i_k.shape)
        print("mean is")
        print(mean_tr.shape)
        print("phi shape")
        print(phi_tr.shape)
        print(phi_tr[:,:k].shape)
        print("b shape")
        print(b.shape)
        print(b[:k].shape)
        p=Points(s_i.reshape(-1,3), c="b", r=6)
        p2=Points(s_i_k.reshape(-1,3), c="g",r=8)
        show(p,p2)
        exit()
      if normalise:
        r_k = r_k_i/max_dist*100.
      else:
        r_k += r_k_i

    r_k /= (n+1)

    return r_k#/max(mean_tr)*100.

  def testGeneralisation(self, testData, mean_tr, phi_tr, k, pca):
    if k < 1:
      print("LOW k found")
      k = np.where(np.cumsum(pca.explained_variance_ratio_)>k)[0][0]
      print("num components is ", k)

    g_k = 0 #initialise error
    for n in range(testData.shape[0]):
      s_i = testData[n].reshape(-1,3) #s = shape
      # normalise = True
      if normalise:
        max_dist = 100.
        # max_dist = cdist(s_i, s_i).max()
        max_dist = abs(s_i.max(axis=0) - s_i.min(axis=0)).max()
      b = np.dot((s_i.reshape(-1) - mean_tr), phi_tr.T)/self.std #-actually b * std, not b
      s_i_k = self.getx_allModes(mean_tr, phi_tr[:k], b[:k]*self.std[:k])

      # g_k_i = np.sum(abs(s_i_k - s_i))/len(s_i) # generalisation error for instance i
      g_k_i = np.sum(utils.euclideanDist(s_i_k, s_i))/len(s_i) # generalisation error for instance i
      if debug:
        print("r_k")
        print(r_k.shape)
        print("SHAPE CHECK")
        print(s_i.shape, s_i_k.shape)
        print("mean is")
        print(mean_tr.shape)
        print("phi shape")
        print(phi_tr.shape)
        print(phi_tr[:,:k].shape)
        print("b shape")
        print(b.shape)
        print(b[:k].shape)
        p=Points(s_i.reshape(-1,3), c="b", r=6)
        p2=Points(s_i_k.reshape(-1,3), c="g",r=8)
        show(p,p2)
        exit()
      if normalise:
        g_k = g_k_i/max_dist*100.
      else:
        g_k += g_k_i

    # print("r_k is ", r_k, "\n n is",n)
    # print("Reconstruction error is", r_k, r_k/max(mean_tr)*100.,"%" )
    g_k /= (n + 1)

    return g_k#/max(mean_tr)*100.

  def testSpecificity(self, trainData, mean_tr, phi_tr, k, pca, N=20):
    '''
      Args:
        Training data (array (samples, 3n)): to compare against
        mean_tr (array (3n)): mean coords for model generation
        phi_tr (array (samples, 3n)): model modes of variation
        k (scalar (float or int)): desired modes or variance
        pca (skLearn object): pca computation
        N (int): number of times to test specificity

      Returns:
        specificity mean error
        specificity standard deviation

      Description:
      Tests specificity of model with k modes of variation.
    '''
    if k < 1:
      print("LOW k found")
      k = np.where(np.cumsum(pca.explained_variance_ratio_)>k)[0][0]
      print("num components is ", k)

    closestList = []
    spec_k = np.zeros(N) #initialise reconstruction error

    for count, n in enumerate(range(N)):

      #-set b to a vector with zero mean and 3 S.D. max
      b = np.random.default_rng().standard_normal(size=(len(phi_tr.T)))
      b = np.where(b>3, 3, b)
      b = np.where(b<-3, -3, b)
      #-generate random shape
      # s_i_k = (mean_tr + np.dot(phi_tr[:,:k], b[:k]))#.reshape(-1,1)
      s_i_k = self.getx_allModes(mean_tr, phi_tr[:k], b[:k]*self.std[:k])

      #-find closest shape in training set
      s_i_k = s_i_k.reshape(-1)
      closestS = np.argmin(np.sum(abs(trainData-s_i_k),axis=1))
      closestList.append(closestS)
      s_i_dash = trainData[closestS]

      #-difference between generated shape and closest match
      spec_k_i = np.sum(abs(s_i_k - s_i_dash))/len(s_i_dash) 

      if debug:
        print("spec_k")
        print(spec_k)
        print("SHAPE CHECK")
        print(s_i_dash.shape, s_i_k.shape)
        print("mean is")
        print(mean_tr.shape)
        print("phi shape")
        print(phi_tr.shape)
        print(phi_tr[:,:k].shape)
        print("b shape")
        print(b.shape)
        print(b[:k].shape)
        p=Points(s_i_dash.reshape(-1,3), c="b", r=6)
        p2=Points(s_i_k.reshape(-1,3), c="g",r=8)
        show(p,p2)
        exit()
      spec_k[count] = spec_k_i
    # spec_k /= max(mean_tr)
    # spec_k *= 100.

    #-return mean and standard deviation
    return np.mean(spec_k), np.std(spec_k) 

  def getx_oneMode(self, xBar, phi, mode=0, dev=3):
    '''
    Return point cloud in which one specific mode of variation has been
    adjusted by a specified deviation
    '''
    b = copy(self.b)
    std = np.sqrt(self.variance)
    b[mode] = dev * std

    return (xBar + np.dot(phi.T, b)).reshape(-1,3)

  def getx_allModes(self, xBar, phi, b):
    '''
    Return point cloud that has been adjusted by a specified shape vector (b)

      Args:
        xBar (3n,) array: mean shape
        phi (sampleNum, 3n) array: PCA components
        b (sampleNum, ) array: shape vector to vary points by
    '''
    return (xBar + np.dot(phi.T, b) ).reshape(-1,3)

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
                      default='allLandmarks/', 
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

  if getModes:
    std_devs = [2, -2]
    modes = [0,1,2]#,3,4,5]
    vp = v.Plotter(N=len(modes))
    '''
    to visualise with graph connectivity, we need to establish correspondence
    between graph nodes and landmarks'''
    baseInd = 0 # use last graph to structure as mean
    lgraphMean = lgraphList[baseInd].copy()
    extraPts = []
    for i, lgraph in enumerate(lgraphList):
      for node in lgraph.nodes:
        pos = lgraph.nodes[node]['pos']

        dist = utils.euclideanDist(nodalCoords[i], pos)
        currentLM = np.argmin(dist)
        isin = np.isclose(posList[i], nodalCoords[i][currentLM]).all(axis=1)
        # print(currentLM)
        # make sure closest coord selected is on skeleton and not surface point
        while isin.sum() == 0:
          if i==0:
            extraPts.append(v.Points([xBar_n3[currentLM]], r=8).c('red'))
          # print(dist.min(), currentLM)
          dist[currentLM] = 100000000
          currentLM = np.argmin(dist)
          # print(dist.min(), currentLM)
          isin = np.isclose(posList[i], nodalCoords[i][currentLM]).all(axis=1)
          # print('\t\t',currentLM)
          if i==0:
            extraPts.append(v.Points([xBar_n3[currentLM]], r=8).c('red'))

        # ind = np.argmin(utils.euclideanDist(nodalCoords[i], pos))
        lgraph.nodes[node]['npID'] = currentLM
        if i==range(0,len(lgraphList))[baseInd]:
          lgraphMean.nodes[node]['pos'] = xBar_n3[currentLM]
          lgraphMean.nodes[node]['npID'] = currentLM
    phi = ssm.phi_s
    
    col = ['blue', 'red']
    root = list(nx.topological_sort(lgraphMean))[0]

    surfs = [None]*len(std_devs)
    surfs = [surfs]*len(modes)
    points = [None]*len(std_devs)
    points = [points]*len(modes)

    date_today = str(date.today())
    surfOutName = "./surfaces/"+date_today+"_mode{}_std{}_mm_.stl"
    for i, mode in enumerate(modes):
      lines = []
      cyl = []
      t = v.Text2D('Mode {}.\nBlue is +3 S.D. Red is -3 S.D'.format(mode+1))
      for j, (sd, c) in enumerate(zip(std_devs,col)):
        b = copy(ssm.b)
        b[mode] = sd
        diff = np.dot(phi.T,b*ssm.std).reshape(-1,3)
        x = xBar_n3 + diff
        xscale = nodalCoords - nodalCoords.mean(axis=0)
        # x *= xscale.std(axis=0)
        # x += nodalCoords.mean(axis=0)
        cyl.append(v.Points(x,r=4).c(c))
        for edge in lgraphMean.edges:
          ind1 = lgraphMean.nodes[edge[0]]['npID']
          p1 = x[ind1]
          ind2 = lgraphMean.nodes[edge[1]]['npID']
          p2 = x[ind2]
          l = v.Line(p1, p2, lw=4).c(c)
          lines.append(l)

        if getModes == 'w':
          lm_template = np.loadtxt(template_lmFile, skiprows=1, delimiter=",",
                                      usecols=[0,1,2])
          carinaTemplate = np.loadtxt(template_lmFileOrig, skiprows=1, delimiter=',',
                                      usecols=[1,2,3])[1]*-1
          mesh_template = v.load(template_meshFile)
          mesh_template = mesh_template.pos(carinaTemplate)
          print(x.shape, lm_template.shape)
          morph = MorphAirwayTemplateMesh(lm_template, x, mesh_template)
          surfs[i][j] = morph.mesh_target.c(c).alpha(0.4)
          points[i][j] = v.Points(x,r=4).c(c)
          v.write(morph.mesh_target, surfOutName)
      if getModes == 'w':
        vp.show(surfs[i], points[i], t, at=mode)
      else:
        vp.show(lines, cyl, t, extraPts, at=mode)
    vp.show(interactive=True)

  if debug:
    if debug == 'mean':
      vp = v.Plotter()
      '''
      to visualise with graph connectivity, we need to establish correspondence
      between graph nodes and landmarks'''
      baseInd = 0 # use last graph to structure as mean
      lgraphMean = lgraphList[baseInd].copy()
      extraPts = []
      lgraph = lgraphList[baseInd].copy()
      for node in lgraph.nodes:
        pos = lgraph.nodes[node]['pos']
        dist = utils.euclideanDist(nodalCoords[i], pos)
        currentLM = np.argmin(dist)
        isin = np.isclose(posList[i], nodalCoords[i][currentLM]).all(axis=1)
        while isin.sum() == 0:
          dist[currentLM] = 100000000
          currentLM = np.argmin(dist)
          isin = np.isclose(posList[i], nodalCoords[i][currentLM]).all(axis=1)
        lgraph.nodes[node]['npID'] = currentLM
        lgraphMean.nodes[node]['pos'] = xBar_n3[currentLM]
        lgraphMean.nodes[node]['npID'] = currentLM
      
      c = 'blue'
      x = xBar_n3
      xscale = nodalCoords - nodalCoords.mean(axis=0)
      x *= xscale.std(axis=0)
      x += nodalCoords.mean(axis=0)
      cyl = []
      lines = []
      cyl.append(v.Points(x,r=4).c(c))
      for edge in lgraphMean.edges:
        ind1 = lgraphMean.nodes[edge[0]]['npID']
        p1 = x[ind1]
        ind2 = lgraphMean.nodes[edge[1]]['npID']
        p2 = x[ind2]
        l = v.Line(p1, p2, lw=4).c(c)
        lines.append(l)
      t = v.Text2D('Mean airway shape')
      vp.show(lines, cyl, t)
      vp.show(interactive=True)
    else:
      col = ["black","blue", "green", "pink", "red", "yellow", "orange",
             "turquoise", "violet", "steelblue", "aqua", "lightgreen", "gold"]
      inds = np.random.randint(0,len(col),nodalCoords.shape[1])
      c = list(np.array(col)[inds])
      # plot all training data connectivity and landmarks    
      vp = v.Plotter(N=len(lgraphList))
      # plot graphs and landmarks with surface mesh for debugging
      if debug == 'm': 
        meshList = [v.load(glob(meshFile.format(cID))[0]) for cID in caseIDs]

      for i, lgraph in enumerate(lgraphList):
        origin = v.Points([np.zeros(3)],r=10, c='green')
        lines = []
        t = v.Text2D('Case {}'.format(caseIDs[i]))
        for edge in lgraph.edges:
          l = v.Line(lgraph.nodes[edge[0]]['pos'], lgraph.nodes[edge[1]]['pos'], 
                      lw=4, c='black')
          lines.append(l)
        pNodes = v.Points(posList[i],r=10,c='black')
        p = v.Points(nodalCoords[i], r=3, c='blue')
        if debug == 'm':
          carina = -nodalCoordsOrig[i][1]
          m = meshList[i].pos(carina[0], carina[1], carina[2])
          vp.show(lines,origin,p,t,m.alpha(0.2), at=i)
        else:
          vp.show(lines,origin,pNodes,t, at=i)
      vp.show(interactive=True)

  if getMetrics:

    compac = []
    reconErr = []
    genErr = []
    specErr= []

    trainSplit = 0.95
    ntrain = int(len(nodalCoords)*trainSplit)-1#[len(nodalCoords)-2, 35-2]
    N = 50 #2#30#5 # number of re-training loops to get statistics
    compac = np.zeros(shape=(ntrain+1, (N)))
    reconErr = np.zeros(shape=(ntrain, N))
    genErr = np.zeros(shape=(ntrain, N))
    specErr = np.zeros(shape=(ntrain, 2)) #-mean and std dev

    # loop first to get cumulative variance statistics ('compactness')
    for i in range(N):
      ssm = RespiratorySSM(nodalCoords, train_size=trainSplit, 
                            quiet=True, normalise=normalise)
      xBar = ssm.computeMean(ssm.x_scale_tr)
      compac[:,i] =  np.cumsum(ssm.pca.explained_variance_ratio_)*100.
    
    # loop for generalisation, reconstruction and specificity errors
    for k in np.arange(1,ntrain+1):
      print("modes =", k)
      for i in range(N):
        ssm = RespiratorySSM(nodalCoords, train_size=trainSplit, 
                             quiet=True, normalise=normalise)
        xBar = ssm.computeMean(ssm.x_scale_tr)

        genErr[k-1,i] = ssm.testGeneralisation(ssm.x_scale_te, 
                                                xBar,
                                                ssm.phi, 
                                                k, 
                                                ssm.pca)
        reconErr[k-1,i] = ssm.testReconstruction(ssm.x_scale_tr, 
                                                  xBar,
                                                  ssm.phi, 
                                                  k, 
                                                  ssm.pca)

      ssm = RespiratorySSM(nodalCoords, train_size=trainSplit, 
                           quiet=True, normalise=normalise)
      xBar = ssm.computeMean(ssm.x_scale_tr)
      
      specErr[k-1,:] = np.array(
                                ssm.testSpecificity(ssm.x_scale_tr, 
                                                    xBar,
                                                    ssm.phi, 
                                                    k, 
                                                    ssm.pca,
                                                    N))
    from ssmPlot import plotSSMmetrics, plotSSMmetrics_three 
    if normalise:
      units = '%'
    else:
      units = 'mm'
    # plotSSMmetrics(compac, reconErr, genErr, specErr, tag="all")#str(shape))
    plotSSMmetrics_three(compac, reconErr, genErr, units=units, 
                         tag="all_noErr_norm")#str(shape))

