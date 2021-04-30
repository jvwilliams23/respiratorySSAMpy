'''
User args given by "python reconstructSSAM.py -h"

    Develops Posterior Shape Model (PSM) of lung lobes based
    on landmarks determined by GAMEs algorithm

    @author: Josh Williams

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
from sklearn.preprocessing import MinMaxScaler

from skimage import io
from skimage.color import rgb2gray
from skimage.morphology import disk
from skimage.filters import rank

from time import time
from datetime import date

# from reconstructSSAM import LobarPSM
from respiratorySAM import RespiratorySAM
from respiratorySSM import RespiratorySSM
from respiratorySSAM import RespiratorySSAM
from morphAirwayTemplateMesh import MorphAirwayTemplateMesh


xRayFile = "interpOut.csv",#"edges-LIDC0002.csv",
#"xRaySeg/output1CannyEdgesCLIPPED1.csv",
tag = "_case0"
template_mesh = "segmentations/template3948/newtemplate3948_mm.stl"
plotAlignment = False

class RespiratoryReconstructSSAM:

  def __init__(self, shape, xRay, lmOrder, normals, transform, 
                density=None, img=None, imgCoords=None,
                model=None, modeNum=None, epochs=200, lRateMax=1000,
                shapePairs=None, 
                c_edge=1.0, c_prior=0.01, c_dense=0.5):

    self.lobes = ['RUL', 'RML', 'RLL', 'LUL', 'LLL']
    self.lmOrder = lmOrder

    self.shape = shape
    self.shapenorms = normals
    self.xRay = xRay #-XR edge map
    self.projLM = None
    self.projLM_ID = None
    self.fissureLM_ID = None
    self.model = model
    self.transform = transform[np.newaxis]
    #-appearance model inputs
    self.density = density #-landmark densities

    '''
    # self.img = ssam.sam.normaliseTestImageDensity(img)
    # self.imgCoords = ssam.sam.drrArrToRealWorld(img,
    #                                             np.zeros(3), 
    #                                             spacing_xr)[0]
    # #-center image coords, so in the same coord system as edges
    # self.imgCoords -= np.array([250,250])*spacing_xr[[0,2]]
    '''
    # format x-ray for image enhancement
    scaler = MinMaxScaler()
    scaler.fit(img)
    img = scaler.transform(img)
    img *= 0.999 #avoid floating point error in scalar causing img > 1
    # img_filt = utils.bilateralfilter(img, 10)
    # if img_filt.max()>1:
    #   img_filt = img_filt/img_filt.max()
    img_local = utils.localNormalisation(img, 10)
    scaler = MinMaxScaler()
    scaler.fit(img_local)
    img_local = scaler.transform(img_local)
    img_local = np.round(img_local,4)
    self.img = img #-XR image array
    print(img.min(), img.max())
    self.imgGrad = rank.gradient(img, disk(5)) / 255
    self.imgCoords = imgCoords #-X and Z coords of X-ray pixels
    self.alignTerm = xRay.mean(axis=0) #-needed for coarse alignment coord frame

    self.scale = 1
    self.translate = np.array([0.,0.])
    self.shapePairs = shapePairs

    self.c_edge = c_edge
    self.c_prior = c_prior
    self.c_dense = c_dense

    self.optIter = 0
    self.optIterSuc = 0
    self.optStage = "align"
    # self.lRate = 1-(np.logspace(0.0, 2, int(round(epochs/10)))
    #               /np.logspace(0.0, 2, int(round(epochs/10))).max())
    self.lRate = np.logspace(2, 0., lRateMax)/100

    self.coordsAll = None
    self.projLM_IDAll = None
    self.fisIDAll = None

    self.optimiseStage = "pose" #-first pose is aligned, then "both"
    # self.eng = 0

    if type(shape) == dict:
      #-initialise shape parameters for each sub-shape, to reduced mode nums
      self.model_s = dict.fromkeys(model.keys())
      self.model_g = dict.fromkeys(model.keys())
      self.b = np.zeros(modeNum)

      for k in model.keys(): 
        #-parameters
        #-shape model components only
        self.model_s[k] = self.filterModelShapeOnly(self.model[k][:modeNum])
        #-gray-value model components only
        self.model_g[k] = self.filterModelDensityOnly(self.model[k][:modeNum])

      self.meanScaled = self.stackShapeAndDensity(
                                                    self.scaleShape(shape['ALL']),
                                                    self.density.mean(axis=0)
                                                    )
    else:
      #-parameters
      self.b = np.zeros(modeNum)
      #-shape model components only
      self.model_s = self.filterModelShapeOnly(self.model[:modeNum])
      #-gray-value model components only
      self.model_g = self.filterModelDensityOnly(self.model[:modeNum])
      self.meanScaled = self.stackShapeAndDensity(
                                                  self.scaleShape(shape),
                                                  self.density.mean(axis=0)
                                                  )

  def objFuncAirway(self, pose, scale=None, b=None):
    #-call initialised variables from __init__
    xRay = self.xRay
    shape = copy(self.shape)
    shapenorms = self.shapenorms
    projLM = self.projLM
    projLM_ID = self.projLM_ID
    fissureLM_ID = self.fissureLM_ID
    # self.transform = np.zeros(self.shape.shape)
    self.optIter += 1
    # pose = dict.fromkeys(shape.keys())
    # scale = dict.fromkeys(shape.keys())

    print("\nNext {0} loop iter {1}".format(self.optimiseStage,self.optIter))

    prior = []
    #-copy mean shape before shape is adjusted
    meanShape = copy(self.shape)['ALL']
    meanAirway = copy(self.shape)['Airway']
    #-call test parameters from optimizer
    self.b = copy(b)
    
    print("\t\t opt params ", pose, scale)
    print("\t\t\t", b)
    #-apply new shape parameters to each lobe
    all_morphed = self.morphAirway(meanShape,#shape[key], 
                                    meanShape.mean(axis=0),  
                                    self.b, 
                                    self.model_s['ALL'][:len(self.b)])
    
    if pose.size == 2:
        pose = np.insert(pose, 1, 0)
    #align = np.mean(shape[key], axis=0)
    align = self.transform
    all_morphed = self.centerThenScale(all_morphed, scale, align)
    #-apply transformation to shape
    all_morphed = all_morphed + pose

    airway_morphed = all_morphed[self.lmOrder['Airway']]
    #-check shape has not moved to be larger than XR or located outside XR
    # outside_bounds = np.any((all_morphed[:,2]>self.imgCoords[:,1].max()) 
    #                         | (all_morphed[:,2]<self.imgCoords[:,1].min())
    #                         | (all_morphed[:,0]>self.imgCoords[:,0].max())
    #                         | (all_morphed[:,0]<self.imgCoords[:,0].min())
    #                         )
    outside_bounds = np.any((all_morphed[:,2]<self.imgCoords[:,1].min())
                            | (all_morphed[:,0]>self.imgCoords[:,0].max())
                            | (all_morphed[:,0]<self.imgCoords[:,0].min())
                            )
    if outside_bounds:
      print("OUTISDE OF BOUNDS")
      return 2 # hard coded, assuming 2 is a large value for loss

    self.scale = scale #-set globally to call in fitTerm
    #-intialise
    keyEdgeDists = 0
    fit = 0
    #-get losses
    lobe_morphed = dict.fromkeys(self.lobes)
    for lobe in self.lobes:
      lobe_morphed[lobe] = all_morphed[self.lmOrder[lobe]]
    fit = self.fitTerm(xRay, lobe_morphed, shapenorms)

    density_t = self.getDensity(all_morphed, 
                                self.img, 
                                self.imgCoords)
    #-TODO - TEST WITH modelled density instead of target?
    shapeIn = self.stackShapeAndDensity(
                                        self.scaleShape(all_morphed),
                                        density_t
                                       )
    # prior = np.sum(abs(b)/self.variance)
    prior = self.priorTerm(shapeIn, 
                           self.meanScaled)

    densityFit = self.densityLoss(density_t,
                                    self.density.mean(axis=0), 
                                    self.model_g['ALL'][:len(self.b)], 
                                    self.b)
    printc("\tfit loss {}\n\tdensity loss {}".format(
              fit,         densityFit))
    printc("\tprior loss", prior)#round(prior,4))
    # self.c_edge = 0.2
    gradFit = self.gradientTerm(airway_morphed, self.imgGrad, self.imgCoords)

    E = 0.3*gradFit+(self.c_prior*prior)+(self.c_dense*densityFit)+(self.c_edge*fit)

    printc("\ttotal loss", E)

    if self.optIter % 100 == 0:
      self.overlayAirwayOnXR(self.img, all_morphed)
    return E

  def gradientTerm(self, coords, imgGrad, imgCoords):
    '''
        Inputs: 
              coords (Nx3 np.ndarray):
              imgGrad (pixel x pixel, np.ndarray):
              imgCoords (pixel x 2 np.ndarray ):
    '''
    lmGrad = self.getDensity(coords, imgGrad, imgCoords)[self.projLM_ID['Airway']]
    return (-1.0*lmGrad).mean()

  def scaleShape(self, shape):
    '''
        return shape (lm x 3 array) with 0 mean and 1 std
    '''
    return (shape-shape.mean(axis=0))/shape.std(axis=0)
    # return (shape-shape.mean(axis=0))/shape.var(axis=0)

  def centerThenScale(self, shape, scale, alignTerm):
    '''
      Center shape and then increase by isotropic scaling. 
      Removes effect of offset on scaling.
    '''
    shape = shape - alignTerm
    shape = shape * scale
    return shape+alignTerm

  def stackShapeAndDensity(self, shape, density):
    '''
        Inputs:
                shape: array (lm x 3)
                density: array (lm x 1)
        Outputs: array(lm x 4)
    '''
    return np.hstack((shape,density[np.newaxis].T))

  def filterModelShapeOnly(self, model):
    '''
      Return model without density in columns.
      Input 2D array, shape = ( nFeature, 4n )
      Return 2D array, shape = ( nFeature, 3n )
      where n = num landmarks
    '''
    #no appearance params
    model_noApp = model.reshape(model.shape[0],-1, 4) 
    return model_noApp[:,:,:-1].reshape(model.shape[0],-1) # -reshape to 2D array

  def filterModelDensityOnly(self, model):
    '''
      Return model without shape in columns.
      Input 2D array, shape = ( nFeature, 4n )
      Return 2D array, shape = ( nFeature, n )
      where n = num landmarks
    '''
    #no shape params
    model_noSh = model.reshape(model.shape[0],-1, 4) 
    return model_noSh[:,:,-1].reshape(model.shape[0],-1) # -reshape to 2D array

  def normaliseDist(self, dist):
    '''
        Normalise a distance or list of distances to have range [0,1]
    '''
    return np.exp(-1*np.array(dist)/5)

  def densityLoss(self, density_t, densityMean, model, b):
    '''

    '''
    #-modelled density
    density_m = self.getg_allModes(densityMean, 
                                    model, 
                                    b*np.sqrt(self.variance)) 
    #-target density (i.e. density at the nearest pixel)
    # density_t = density_t #self.getDensity(lm, img, imgCoords)
    return np.sum(abs(density_t-density_m))/density_t.shape[0]


  def morphShape(self, shape, transform, shapeParams, model):
    '''
    Adjust shape transformation and scale. 
    Imports to SSM and extracts adjusted shape.
    '''
    removeTrans = shape - transform
    removeMean = removeTrans.mean(axis=0)
    shapeCentre = removeTrans - removeMean
    scaler = shapeCentre.std(axis=0)
    shapeSc = shapeCentre/scaler #StandardScaler().fit_transform(shapeCentre)

    # shapeOut = shapeSc + np.dot(model.T, #-04/11/20 - test diff normalisation
    #                             shapeParams).reshape(-1,3)
    shapeOut = shapeSc + np.dot(shapeParams[np.newaxis,:]
                                  *np.sqrt(self.variance[np.newaxis,:]),
                                model).reshape(-1,3)

    shapeDiff = np.sqrt(np.sum((shapeOut-shapeSc)**2, axis=1))
    print("shape diff [normalised] \t mean", np.mean(shapeDiff), 
          "\t max", np.max(shapeDiff))
    # shapeOut = ssam.getx_allModes(shapeSc.reshape(-1), 
    #                                 model, 
    #                                 shapeParams)

    # shapeOut = ((shapeOut*shapeSc.std(axis=0)
    #             +shapeSc.mean(axis=0))
    #             *scaler) \
    #             +removeMean+transform
    shapeOut = (shapeOut
                *scaler) \
                +removeMean+transform


    shapeDiff = np.sqrt(np.sum((shapeOut-shape)**2, axis=1))
    print("shape diff [real space] \t mean", np.mean(shapeDiff), 
          "\t max", np.max(shapeDiff))

    return shapeOut

  def morphAirway(self, shape, transform, shapeParams, model):
    '''
    Adjust shape transformation and scale. 
    Imports to SSM and extracts adjusted shape.
    '''
    removeTrans = shape - transform
    removeMean = removeTrans.mean(axis=0)
    shapeCentre = removeTrans - removeMean
    scaler = shapeCentre.std(axis=0)
    shapeSc = shapeCentre/scaler #StandardScaler().fit_transform(shapeCentre)

    # shapeOut = shapeSc + np.dot(model.T, #-04/11/20 - test diff normalisation
    #                             shapeParams).reshape(-1,3)
    shapeOut = shapeSc + np.dot(shapeParams[np.newaxis,:]
                                  *np.sqrt(self.variance[np.newaxis,:]),
                                model).reshape(-1,3)

    shapeDiff = np.sqrt(np.sum((shapeOut-shapeSc)**2, axis=1))
    print("shape diff [normalised] \t mean", np.mean(shapeDiff), 
          "\t max", np.max(shapeDiff))
    # shapeOut = ssam.getx_allModes(shapeSc.reshape(-1), 
    #                                 model, 
    #                                 shapeParams)

    # shapeOut = ((shapeOut*shapeSc.std(axis=0)
    #             +shapeSc.mean(axis=0))
    #             *scaler) \
    #             +removeMean+transform
    shapeOut = (shapeOut
                *scaler) \
                +transform #+removeMean


    shapeDiff = np.sqrt(np.sum((shapeOut-shape)**2, axis=1))
    print("shape diff [real space] \t mean", np.mean(shapeDiff), 
          "\t max", np.max(shapeDiff))

    return shapeOut

  def optimiseAirwayPoseAndShape(self,objective,init,bounds,epochs=2,threads=1):
    '''
        Minimises objective function using Nevergrad gradient-free optimiser
    '''
    instrum = ng.p.Instrumentation(
                                    pose=ng.p.Array( #shape=(3,), 
                                                init=init[:2]).\
                                        set_bounds(bounds[:2,0], 
                                                   bounds[:2,1]),
                                    scale=ng.p.Scalar(#Scalar(
                                                init=init[2]).\
                                        set_bounds(bounds[2,0],
                                                   bounds[2,1]),
                                    b=ng.p.Array(
                                                init=np.zeros(self.b.size)).\
                                        set_bounds(bounds[3,0],
                                                   bounds[3,1])
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
            x = optimizer.ask()
            loss = objective(*x.args, **x.kwargs)
            optimizer.tell(x, loss)
            lossLog.append(loss)
    recommendation = optimizer.provide_recommendation() #-update recommendation
        
    tag = ''
    utils.plotLoss(lossLog, stage=self.optimiseStage) #-plot loss


    optOut = dict.fromkeys(["pose","scale","b"])
    # optOut = dict.fromkeys(["pose","b"])
    optOut["pose"] = recommendation.value[1]["pose"]
    optOut["scale"] = recommendation.value[1]["scale"]
    optOut['b'] = recommendation.value[1]['b']
    print('recommendation is', recommendation.value)

    return optOut

  def combineAllLobes(self, shape, projLM_ID, fissureLM_ID):
    keys = list(shape.keys())

    for k, key in enumerate(keys):
        if k == 0:
            coordsAll = shape[key]
            fisIDAll = fissureLM_ID[key]
            projLM_IDAll = projLM_ID[key]
        else:
            #-create 2D array with points from all lobes
            coordsAll = np.vstack((coordsAll, shape[key]))
            #-make index of fissure coords consistent with shapeArr
            fisIDAll_tot = list(
                                np.array(fissureLM_ID[key])\
                                +len(shape[keys[k-1]])
                                )
            fisIDAll = fisIDAll + fisIDAll_tot
            #-make index of projection LMs coords consistent with shapeArr
            projLM_IDAll_tot = list(
                                    np.array(projLM_ID[key])\
                                    +len(shape[keys[k-1]])
                                    )
            projLM_IDAll = projLM_IDAll + projLM_IDAll_tot

    return coordsAll, projLM_IDAll, fisIDAll

  def fitTerm(self, xRay, shapeDict, pointNorms3DDict):

    v = 5. #distance weighting factor
    n = 0 #initialise number of points
    thetaList = []# * len(shapeDict.keys())

    plt.close()
    plt.plot(xRay[:,0], xRay[:,1], lw=0, marker="o", ms=2, c="black")

    for k, key in enumerate(self.lobes):

      #-get relative distance between fissure landmarks
      # fd = 1
      # if self.optimiseStage=="both":
      #     for pair in self.shapePairs[key]:
      #         fdFull = cdist(shapeDict[key][self.fissureLM_ID[key]],
      #                         shapeDict[pair][self.fissureLM_ID[pair]],
      #                         metric="euclidean")

      #         #-make mahalanobis dist between fissures relative to initial dist
      #         relFissDist = abs((self.baseFissDists[key][pair]
      #                           -np.mean(np.min(fdFull,axis=1)) ))
      #         fd -= np.exp(-relFissDist/5)#**0.5 

      #     print("\trelFissDist {0}, fd term is {1}".format(
      #                                               round(relFissDist,5),
      #                                               round(fd,5))
      #             )
      #-get only fd term for RML
      if key != "RML":

        shape = shapeDict[key][self.projLM_ID[key]][:,[0,2]]
        # pointNorms3D = pointNorms3DDict[key][self.projLM_ID[key]]

        d_i = np.zeros(shape.shape[0])
        n += len(shape)
        if len(xRay.shape) > 2: #-if xRay is a 3D array
            n_proj = xRay.shape[2]
        else:
            n_proj = 1
        theta = np.zeros(shape.shape[0])

        #-get distance term (D_i)
        distArr = cdist(shape, xRay)
        d_i = np.min(distArr, axis=1)
        D_i = np.exp(-d_i/v) #np.sqrt(np.exp(-d_i/v))

        ''' 
        closestEdge = np.argmin(distArr, axis=1)
        copyArr = np.copy(distArr)
        # hard code min to high value to allow next min to give 2nd lowest
        copyArr[:,closestEdge] = np.inf 
        closestEdge_sec = np.argmin(copyArr, axis=1)

        #-get orientation term (w_i)
        #-get normal vector of two closest points
        normProj2D = (xRay[closestEdge] - xRay[closestEdge_sec])[:,::-1]
        normProj2D = np.where(normProj2D==np.zeros(2), 
                              np.ones(2), 
                              normProj2D)
        #-factor to normalise vector
        div = np.sqrt(np.einsum('ij,ij->i', normProj2D, normProj2D)) 
        normProj2D = np.divide(normProj2D, np.c_[div, div])

        normProj3D = np.insert(normProj2D, 1, 0, axis=1) # set y index to 0
        #-adjust point normals for anisotropic surface scaling
        normLM3D = pointNorms3D/self.scale#np.insert(self.scale[key], 1, 1)
        div = np.sqrt(np.einsum('ij,ij->i', normLM3D, normLM3D)) 
        normLM3D = np.divide(normLM3D, np.c_[div, div, div])

        #-do not need to divide by mag as should be unit norms => = 1 
        alpha = abs(
                    np.arccos( 
                            np.einsum('ij,ij->i', normProj3D, normLM3D) 
                             )*180/pi
                    )
        w_i = np.where( alpha>90, 0, np.cos(alpha*pi/180)  )
        theta = abs( 1 - ( D_i * w_i ) )
        # theta = abs( 1 - ( D_i * w_i * fd) )
        ''' 
        theta = abs( 1 - D_i )
        

        thetaList.append( np.sum(theta) )
    
        # plt.plot(shapeDict[key][np.where(( theta!=0 ))[0],0], 
        #         shapeDict[key][np.where(( theta!=0 ))[0],2], 
        #         lw=0, marker="o", ms = 2)
      # else:
        # # pass
        # # #-n for RML only uses fissure LMs
        # n += len(self.fissureLM_ID[key]) 
        # theta = abs( 1 - (fd) )

        # thetaList.append( np.sum(theta) )
      # del fd

    
    E_fit = ( 1 / ( n * n_proj) ) * np.sum(thetaList)
    
    return E_fit

  def priorTerm(self, shape, meanShape):
    '''
    Compare shape generated by optimisation of shape parameters 
    with mean shape of dataset, using mahalanobis distance.
    '''
    
    #-centre shape
    # meanShapeAligned = meanShape-np.mean(meanShape, axis=0)
    # shapeAligned = shapeDict-np.mean(shapeDict, axis=0)

    #-get avg Mahalanobis dist from mean shape
    # E_prior = np.mean(utils.mahalanobisDist(meanShape, 
    #                                         shape)
    #                     )
    E_prior = np.sum(utils.mahalanobisDist(meanShape, 
                                            shape)
                        )/meanShape.shape[0]

    return E_prior

  def overlayAirwayOnXR(self, img, coords):
    extent=[-self.img.shape[1]/2.*self.spacing_xr[0],  
              self.img.shape[1]/2.*self.spacing_xr[0],  
              -self.img.shape[0]/2.*self.spacing_xr[2],  
              self.img.shape[0]/2.*self.spacing_xr[2] ]
    plt.close()
    plt.imshow(img, cmap='gray', extent=extent)
    plt.scatter(coords[:,0], coords[:,2],s=2,c='black')
    plt.savefig('images/reconstruction/debug/iter{}.png'.format(str(self.optIter)))
    # plt.show()
    # exit()
    return None

  def searchLevelThree(self, img, imgCoords):
    return img[::8, ::8], imgCoords[::8]

  def searchLevelTwo(self, img, imgCoords):
    return img[::2,img::2], imgCoords[::2]

  def getProjectionLandmarks(self, faceIDs, faceNorms, points):
    '''
    args:
        faceIDs array(num faces, 3): Each row has three IDs corresponding 
                to the vertices that construct that face.
        faceNorms array(num faces, 3): components of each face normal vector
        points array(num points, 3): coordinates of each surface point
    returns:
        projectionLM array(n, 2): 2D projection coordinates of silhouette landmarks
    '''
    assert type(points)==type(faceIDs)==type(faceNorms), \
      'type mismatch in surface faces, normals and points'

    if type(points)==dict:
      projectionLM = dict.fromkeys(points.keys())
      projectionLM_ID = dict.fromkeys(points.keys())

      for shape in points.keys():
        norms = []
        projectionLM[shape] = []
        projectionLM_ID[shape] = []
        for pID in range(len(points[shape])):
          norms.append( np.where( (faceIDs[shape][:,0]==pID) \
                                  | (faceIDs[shape][:,1]==pID) \
                                  | (faceIDs[shape][:,2]==pID)
                                )[0]
                      )
          if len(norms[pID]) > 1:
            '''check if y normal for point has +ve and -ve components
               in the projection plane '''
            if np.min(faceNorms[shape][norms[pID]][:,1]) < 0 \
            and np.max(faceNorms[shape][norms[pID]][:,1]) > 0:
                projectionLM[shape].append(points[shape][pID])
                projectionLM_ID[shape].append(pID)
          else:
              continue
        projectionLM[shape] = np.array(projectionLM[shape])
        #-delete projection plane from coords
        projectionLM[shape] = np.delete(projectionLM[shape], 1, axis=1)
        projectionLM_ID[shape] = np.array(projectionLM_ID[shape])
    else:
      norms = []
      projectionLM = []
      projectionLM_ID = []
      for pID in range(len(points)):
        norms.append( np.where( (faceIDs[:,0]==pID) \
                                | (faceIDs[:,1]==pID) \
                                | (faceIDs[:,2]==pID)
                              )[0]
                    )
        if len(norms[pID]) > 1:
          '''check if y normal for point has +ve and -ve components
             in the projection plane '''
          if np.min(faceNorms[norms[pID]][:,1]) < 0 \
          and np.max(faceNorms[norms[pID]][:,1]) > 0:
              projectionLM.append(points[pID])
              projectionLM_ID.append(pID)
        else:
            continue
      ids = np.arange(0, len(points))
      np.where( np.isin(faceIDs[:,0], points) \
                                | (faceIDs[:,1]==pID) \
                                | (faceIDs[:,2]==pID)
                              )[0]

      '''
      projectionLM_ID = []
      for pID in range(len(points)):
        norms.append( np.where( (faceIDs[:,0]==pID) \
                                | (faceIDs[:,1]==pID) \
                                | (faceIDs[:,2]==pID)
                              )[0]
                    )
        if len(norms[pID]) > 1:
          if np.min(faceNorms[norms[pID]][:,1]) < 0 \
          and np.max(faceNorms[norms[pID]][:,1]) > 0:
              projectionLM_ID.append(pID)
        else:
            continue
      '''
      projectionLM = np.array(projectionLM)
      #-delete projection plane from coords
      projectionLM = np.delete(projectionLM, 1, axis=1)        
      projectionLM_ID[shape] = np.array(projectionLM_ID[shape])
    return projectionLM, projectionLM_ID

  def saveSurfProjectionComparison(self, E, xRay):
    print("\n\tFIT IS", E)

    plt.text(xRay[:,0].min()*0.9,xRay[:,1].max()*0.9, 
             "E = {0}".format(round(E,6)))
    plt.savefig("images/xRayRecon/nevergrad/"
                +self.optimiseStage
                +str(self.optIter)
                +".png")
    return None

  def deleteShadowedEdges(self,coords,projLM,projLM_ID):
    '''
    Deletes landmarks that are not found on the radiograph 
    (overlapped by spine or fissures that are not visible)
    '''
    # if "RUL" in coords.keys()\
    # and "RLL" in coords.keys():
    #   lpsm.shape = { filterKey: coords[filterKey] \
    #               for filterKey in ["RUL","RLL"] }
    # else:
    shape = copy(coords) #-why?
    
    widthMaxCutoff = 0.5
    if "RUL" in projLM.keys():
        upperKey = "RUL"
        heightMax = 0.8
    else:
        upperKey = "RLL"
        heightMax = 1.3
    height = projLM[upperKey][:,1].max()\
             -projLM["RLL"][:,1].min()
    for key in coords.keys():
      print(key)
      # if key[0] == "L":
      #   continue
      delInd = []
      if key == "RML" or key == 'Airway':
        projLM_ID[key] = np.array(projLM_ID[key])
        continue
      if key == "RLL":
          heightMin = -0.01
          widthCutoff = 0.65
          widthMaxCutoff = 0.5#2
      else:#if key == "RLL":
          heightMin = -0.01#0.3
          widthCutoff = 0.85
          widthMaxCutoff = 1
      width = projLM[key][:,0].max()\
              -projLM[key][:,0].min()
      if key != "LLL":
        delInd = np.where( 
                           (projLM[key][:,0]\
                           -projLM[key][:,0].min()
                           >widthCutoff*width) 
                           | 
                           ((projLM[key][:,1]\
                           -projLM["RLL"][:,1].min()
                           >heightMin*height) & 
                           (projLM[key][:,1]\
                           -projLM["RLL"][:,1].min()
                           <heightMax*height) & 
                           (projLM[key][:,0]\
                           -projLM[key][:,0].min()
                           >0.2*width) & 
                           (projLM[key][:,0]\
                           -projLM[key][:,0].min()
                           <widthMaxCutoff*width) # JW 26/06/20
                           ) )[0]
      if key=="RUL":
        #-filter low points 
        #-(corresponds to curvature at RUL RLL intersection)
        delInd = np.unique(
                 np.append(delInd,
                 np.where(
                            projLM[key][:,1]
                            <((projLM[key][:,1].max()
                                -projLM[key][:,1].min())*0.2)
                                +projLM[key][:,1].min()  
                          )[0]
                          )
                          )

        projLM[key] = np.delete(projLM[key], delInd, axis=0)
        projLM_ID[key] = np.delete(projLM_ID[key], delInd)
        del delInd #-delete to avoid accidental deletion of wrong indexes

        '''filter outlier clusters by applying 75th percentile 
            mahalanobis distance filter twice'''
        tmpprojLM = copy(projLM[key]) #-store initial values
        for i in range(2):
            #-keep data to outer side of RUL
            dataKept = projLM["RUL"][np.where(projLM["RUL"][:,0]
                                          <(projLM["RUL"][:,0].min()
                                            +(projLM["RUL"][:,0].max()
                                              -projLM["RUL"][:,0].min() )/2)
                                          )
                                ]
            #-set aside inner coordinates for filtering
            data = projLM["RUL"][np.where(projLM["RUL"][:,0]
                                          >(projLM["RUL"][:,0].min()
                                            +(projLM["RUL"][:,0].max()
                                            -projLM["RUL"][:,0].min() )/2)
                                          )
                                ]
    
            md = cdist(data, data, "mahalanobis")
            #-set coords with high mean mahalanobis distance for deletion
            delInd = np.unique( 
                                np.where(md.mean(axis=0)
                                        >np.percentile(md.mean(axis=0), 65))[0]
                                )

            data = np.delete(data, delInd, axis=0)
            del delInd #-delete to avoid accidental deletion of wrong indexes
            projLM[key] = np.vstack((dataKept,data)) #-reset array
        #-loop to find landmark ID's removed
        idkept = []
        for lmID in projLM_ID["RUL"]:
            if tmpprojLM[np.where(projLM_ID["RUL"]==lmID)] in projLM["RUL"]:
                idkept.append(lmID)
        projLM_ID["RUL"] = np.array(idkept)
      elif key == "RLL":
        '''
        delete upper half of landmark points (due to fissure overlap)
        OR
        inner fifth (0.2) of landmarks (not visible on X-ray)
        '''
        delInd = np.where(
                           (
                            (projLM["RLL"][:,1]\
                           -projLM["RLL"][:,1].min()
                           >0.6*height) 
                           |
                           ((projLM["RLL"][:,0]\
                           -projLM["RLL"][:,0].min()
                           >0.8*width) )
                           )
                         )

        #-delete previous stored indexes
        projLM[key] = np.delete(projLM[key], delInd, axis=0)
        projLM_ID[key] = np.delete(projLM_ID[key], delInd)

        #-filter out upper right points in the RLL 
        #-set coords with high mean mahalanobis distance for deletion
        delInd = np.unique( 
                            np.where(md.mean(axis=0)
                                    >np.percentile(md.mean(axis=0), 92.5))[0]
                            )
        projLM[key] = np.delete(projLM[key], delInd, axis=0)
        projLM_ID[key] = np.delete(projLM_ID[key], delInd)

        for i in range(2):
            #-find lowest point in RLL to filter out
            #-this is needed as it is typically not segmented by XR seg tool
            cornerP = projLM["RLL"][np.argmin(projLM["RLL"][:,1], axis=0)]
            cutoff = 0.4 #-define cutoff distance

            #-use mahalanobis distance to filter out points
            projLM_ID["RLL"] = projLM_ID["RLL"][np.where(
                                        utils.mahalanobisDist(projLM["RLL"],
                                                              cornerP)
                                                              >cutoff)]
            projLM["RLL"] = projLM["RLL"][np.where(
                                          utils.mahalanobisDist(projLM["RLL"],
                                                                cornerP)
                                                                >cutoff)]
      elif key == "LUL": #-if left lung
        delInd = np.where( ((projLM[key][:,0]-projLM[key][:,0].min()
                              <0.4*width)
                              &
                              (projLM[key][:,1]-projLM[key][:,1].min()
                              <0.8*height))
                            |
                            (projLM[key][:,1]-projLM[key][:,1].min()
                            <0.4*height) )
        projLM[key] = np.delete(projLM[key], delInd, axis=0)
        projLM_ID[key] = np.delete(projLM_ID[key], delInd)
        # elif key == "LLL": #-if left lung
        #   delInd = np.where( ((projLM[key][:,0]-projLM[key][:,0].min()
        #                       <0.7*width)
        #                       &
        #                       (projLM[key][:,1]-projLM[key][:,1].min()
        #                       >0.4*height))
        #                       |
        #                       (projLM[key][:,0]-projLM[key][:,0].min()
        #                       <0.4*width)
        #                       |
        #                       (projLM[key][:,1]-projLM[key][:,1].min()
        #                       <0.1*height)
        #                       )
        #   projLM[key] = np.delete(projLM[key], delInd, axis=0)
        #   projLM_ID[key] = np.delete(projLM_ID[key], delInd)
          
        #   md = cdist(projLM[key], projLM[key], "mahalanobis")
        #   #-set coords with high mean mahalanobis distance for deletion
        #   delInd = np.unique( 
        #                       np.where(md.mean(axis=0)
        #                               >np.percentile(md.mean(axis=0), 92.5))[0]
        #                       )
        #   projLM[key] = np.delete(projLM[key], delInd, axis=0)
        #   projLM_ID[key] = np.delete(projLM_ID[key], delInd)
      else:
          projLM[key] = np.delete(projLM[key], delInd, axis=0)
          projLM_ID[key] = np.delete(projLM_ID[key], delInd)
    return projLM, projLM_ID  


def getInputs():
  parser = argparse.ArgumentParser(description='SSM for lung lobe variation')
  parser.add_argument('--inp', '-i',
                      default=False, 
                      type=str, required=True,
                      help='input files (landmarks)'
                      )
  parser.add_argument('--case', '-c',
                      default=False, 
                      type=str,#, required=True,
                      help='training data case'
                      )
  parser.add_argument('--out', '-o',
                      default=False, 
                      type=str, required=True,
                      help='output surface tag '
                      )
  parser.add_argument('--var', '-v',
                      default=0.7, 
                      type=float, 
                      help='fraction of variance in training set to include in model [0,1]'
                      )
  parser.add_argument('--c_prior', '-cp',
                      default=0.025, 
                      type=float, 
                      help='prior shape loss coefficient'
                      )
  parser.add_argument('--c_dense', '-cd',
                      default=0.25, 
                      type=float, 
                      help='density loss coefficient'
                      )
  parser.add_argument('--c_edge', '-ce',
                      default=1.0, 
                      type=float, 
                      help='edge map loss coefficient'
                      )
  parser.add_argument('--drrs', 
                      default=False, 
                      type=str, required=True,
                      help='input files (drr)'
                      )
  parser.add_argument('--meshdir', '-m',
                      default=False, 
                      type=str, required=True,
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
                      default=500, 
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

def getScaledAlignedLMs(coords, scale, transform, pose, outlineIDs ):
  scaled = (lpsm.centerThenScale(coords, 
                                 scale,
                                 transform)\
            + np.insert(pose, 1, 0))
  return scaled, scaled[outlineIDs]

def allLobeDensityError(meanScaled, densityOut, densityMean=0, tag=""):
  '''
    Plots the error in gray-value of reconstruction compared to the mean
  '''
  plt.close()
  fig, ax = plt.subplots(nrows=1,
                        ncols=len(meanScaled.keys()), 
                        figsize=(16/2.54,10/2.54))

  for i, key in enumerate(meanScaled.keys()):
    if len(meanScaled.keys()) == 1:
      ax_set = ax
    else:
      ax_set = ax[i]
    xbar = meanScaled[key][:,[0,1,2]]
    gbar = densityMean #meanScaled[key][:,-1]
    gout = densityOut[key]
    a = ax_set.scatter(xbar[:,0], xbar[:,2], 
                      cmap="seismic", 
                      c=abs(gbar-gout).reshape(-1),
                      vmin=0, vmax=1,
                      s=1)
    ax_set.axes.xaxis.set_ticks([])
    ax_set.axes.yaxis.set_ticks([])
    ax_set.set_title(str(key), fontsize=12)
  #-set colorbar
  if len(meanScaled) == 1:
    cb = fig.colorbar(a, ax=ax)
  else:
    cb = fig.colorbar(a, ax=ax.ravel().tolist())
  cb.set_label("density", fontsize=11)
  fig.suptitle("Density error in reconstruction", fontsize=12)
  # plt.show()
  fig.savefig("./images/xRayRecon/nevergrad/density-error"+tag+".png", 
                pad_inches=0, format="png", dpi=300)
  return None

def visualiseOutput(coords, scale=[1,1,1], pose=[0,0,0]):
    #visualiseOutput(surfCoords_mm, optTrans["scale"], optTrans["pose"])
    v = []
    for key in coords.keys():
        v.append(v.Points((coords[key]*scale[key]) + pose[key], r=4))
        np.savetxt(key+"coords.txt", 
                    (coords[key]*scale[key]) + pose[key], 
                    delimiter=",",
                    header="x,y,z")
    v.show(v[0],v[1],v[2])
    return None

def loadXR(file):
  '''

  TODO MAYBE NEED TO NORMALISE???????????????????
  IT IS NORMALISED IN USERUTILS.PY
  '''
  g_im = rgb2gray(io.imread(file))
  g_im[0] = g_im[1].copy()
  g_im = utils.he(g_im[::imgSpaceCoeff,::imgSpaceCoeff])
  return g_im

def registered_output(coords, outline, img, optStage, tag="", density=None):
  plt.close()
  plt.imshow(img, extent=extent, cmap="gray") 
  plt.scatter(outline[:,0], outline[:,1], s = 1, color="black")

  for key in coords.keys():
    if density:
      plt.scatter(coords[key][:,0], coords[key][:,2], s = 1, 
                  c=density[key], cmap="gray", vmin=-1, vmax=1,)
    else:
      plt.scatter(coords[key][:,0], coords[key][:,2], s = 1, c="yellow")
  plt.text(edgePoints_mm[:,0].min()*0.9,edgePoints_mm[:,1].max()*1.2, 
           "trans "+str(optStage["pose"])+"\nscale "+str(optStage["scale"]) )
  plt.savefig("images/xRayRecon/nevergrad/"
              +lpsm.optimiseStage
              +"Final"+tag+".png")
  return None

if __name__ == "__main__":
  date_today = str(date.today())
  print(__doc__)
  startTime = time()

  landmarkDir, case, tag, describedVariance, drrDir, debug, \
          shapes, surfDir, numEpochs, xrayEdgeFile, \
          c_edge, c_dense, c_prior, imgSpaceCoeff = getInputs()
  img=None
  spacing_xr=None
  template_lm = landmarkDir+"/landmarks0{}-case8684.csv"

  # imgSpaceCoeff = 2

  #-check to see if png file is in directory
  # expectedImgName = xrayEdgeFile.replace("outline-","").replace(".csv", ".png")
  # if len(glob(expectedImgName)) !=0:
  #   if len(glob(expectedImgName)) != 1:
  #     printc("when reading XR to match to, too many files found",
  #             expectedImgName, c="r")
  #     exit()
  #   else:
  #     img = loadXR(glob(expectedImgName)[0])
  #     spacing_xr = np.loadtxt(glob(expectedImgName.replace(".png",".md"))[0], 
  #                             skiprows=1)*imgSpaceCoeff
  #     #-FOR CNN X-ray edge map, must include below due to issue when we wrote spacing
  #     # spacing_xr *= (512/500)

  # #-fraction of variance in training set to include in model
  # describedVariance = 0.9#0.9

  #-dict of lobes and adjacent lobes (used for getting fissures)
  shapePairs = {
                "RUL": ["RML", "RLL"],
                "RML": ["RUL", "RLL"],
                "RLL": ["RUL", "RML"],
                "LUL": ["LLL"],
                "LLL": ["LUL"]
               }
  if "ALL" in shapes:
    subshapes = ["RUL", "RML","RLL", "LUL", "LLL"]
  else:
    subshapes = copy(shapes)

  landmarks = dict.fromkeys(subshapes)
  lmTrans_total = dict.fromkeys(subshapes)
  surfCoords = dict.fromkeys(subshapes)
  surfCoords_mm = dict.fromkeys(subshapes)
  surfCoords_mmOrig = dict.fromkeys(subshapes)
  meanNorms_points = dict.fromkeys(subshapes)
  numModes = dict.fromkeys(subshapes)
  #-dicts for projection landmarks
  meanNorms_face = dict.fromkeys(subshapes)
  faces = dict.fromkeys(subshapes)
  model = dict.fromkeys(subshapes) #-shape model 
  #-dicts for appearance model inputs
  lmProj = dict.fromkeys(subshapes)
  drrArr = dict.fromkeys(subshapes)
  origin = dict.fromkeys(subshapes)
  spacing = dict.fromkeys(subshapes)
  lmTrans_all = dict.fromkeys(subshapes)
  #-dicts for appearance model outputs
  drrPos = dict.fromkeys(subshapes)
  # drrArrNorm = dict.fromkeys(subshapes)
  density = dict.fromkeys(subshapes)
  #-save reordering of surface points to landmark correspondance
  surfToLMorder = dict.fromkeys(subshapes)

  print("\tReading data")
  for shape in subshapes:
    print("\t ",shape)
    #-read DRR data
    originDirs = glob( drrDir + "/origins/origins/drr*.md")#.sort()
    spacingDirs = glob( drrDir + "/*/drr*.md")#.sort()
    imDirs = glob(drrDir + "/*/drr*.png")#.sort()
    originDirs.sort()
    spacingDirs.sort()
    imDirs.sort()

    patientIDs = [i.split("/")[-1].replace(".png", "")[-4:] for i in imDirs]
    landmarkDirs = glob( landmarkDir+"/landmarks*{0}*.csv".format(shape) )
    landmarkDirs = sorted(landmarkDirs, 
                          key=lambda x: int(x.replace(".csv","")[-4:]))
    lmIDs = [i.split("/")[-1].replace(".csv", "")[-4:] for i in landmarkDirs]
    transDirs = glob( "savedPointClouds/allLandmarks/"
                                +"transformParams_case*"
                                +"_m_"+shape+".dat")
    transDirs_all = glob( "savedPointClouds/allLandmarks/"
                                +"transformParams_case*"
                                +"_m_"+shape+".dat")
    transDirs = sorted(transDirs, 
                          key=lambda x: int(x.split("case")[1][:4]))
    transDirs_all = sorted(transDirs_all, 
                          key=lambda x: int(x.split("case")[1][:4]))
    transIDs = [i.split("_case")[1].split("_m_")[0] for i in transDirs_all]
    #-remove scans without landmarks from DRR dirs
    missing = []
    missingID = []
    # exit()
    for p, pID in enumerate(patientIDs): 
      if pID not in lmIDs: 
        missing.append(p)
        missingID.append(pID)
    print("Missing {} datasets. Dataset reduced from {} to {}".format(len(missing),
                                                                      len(patientIDs),
                                                                      len(lmIDs)))
    '''loop in reverse so no errors due to index being deleted 
        i.e. delete index 12, then next item to delete is 21, 
        but beca  use a previous element has been removed, you delete item 20'''
    for m in missing[::-1]:
      patientIDs.pop(m)
      # landmarkDirs.pop(m)
      originDirs.pop(m)
      spacingDirs.pop(m)
      imDirs.pop(m)
      # transDirs_all.pop(m)
      # transDirs.pop(m) #-commented as these have been deleted manually
    missing = []
    missingID = []
    for p, pID in enumerate(patientIDs): 
      if pID not in transIDs: 
        missing.append(p)
        missingID.append(pID)
    '''loop in reverse so no errors due to index being deleted 
        i.e. delete index 12, then next item to delete is 21, 
        but beca  use a previous element has been removed, you delete item 20'''
    for m in missing[::-1]:
      landmarkDirs.pop(m)
      patientIDs.pop(m)
      originDirs.pop(m)
      spacingDirs.pop(m)
      imDirs.pop(m)

    # lmTrans_total[shape] = np.loadtxt("savedPointClouds/"
    #                                       +"transformParams"+shape+".dat", 
    #                                        skiprows=1, max_rows=1)
    lmTrans_total[shape] = np.loadtxt("savedPointClouds/"
                                          +"transformParams_m_"+shape+".dat", 
                                           skiprows=1, max_rows=1)
    lmTrans_all[shape] = np.vstack([np.loadtxt(t, skiprows=1, max_rows=1) 
                                    for t in transDirs_all])

    # landmarkFiles = glob(landmarkDir+"landmark*"+shape+"*")
    # landmarkFiles = sorted(landmarkFiles, 
    #                       key=lambda x: int(x.split("case")[1][:4]))
    landmarks[shape] = np.array([np.loadtxt(l, delimiter=",") 
                                  for l in landmarkDirs])
    landmarks[shape] = landmarks[shape]#[:,::4,:]

    #-read appearance modelling data
    origin[shape] = np.vstack([np.loadtxt(o, skiprows=1)] for o in originDirs)
    spacing[shape] = np.vstack([np.loadtxt(o, skiprows=1)*imgSpaceCoeff] 
                                for o in spacingDirs)
    drrArr[shape] = np.rollaxis(
                        np.dstack([loadXR(o) for o in imDirs]),
                        2, 0)
    #-offset centered coordinates to same reference frame as CT data
    lmProj[shape] = landmarks[shape] + lmTrans_all[shape][:, np.newaxis]
  
    #-load pre-prepared mean stl and point cloud of all data
    mean_shape = surfDir+'totmean'+shape+'.stl'
    # meanArr = np.loadtxt("savedPointClouds/totmeanCloud"+shape+".xyz")#[::4]
    # -meanArr = ssam.xg_train
    meanArr = landmarks[shape].mean(axis=0)

    #-load mesh data and create silhouette
    mesh = v.load(mean_shape).computeNormals()

    #-extract mesh data (coords, normals and faces)
    surfCoords[shape] = mesh.points()
    meanNorms_points[shape] = mesh.normals(cells=False)
    meanNorms_face[shape] = mesh.normals(cells=True)
    faces[shape] = np.array(mesh.faces())

    #-offset to ensure lobes are stacked 
    surfCoords_mm[shape] = surfCoords[shape] + lmTrans_total[shape] 
    surfCoords_mmOrig[shape] = copy(surfCoords_mm[shape])
    #-reorder unstructured stl file to be coherent w/ model and landmarks
    surfToLMorder[shape] = []
    for point in meanArr:
      surfToLMorder[shape].append( np.argmin( utils.euclideanDist(surfCoords[shape], 
                                                                  point) ) )
    surfCoords_mm[shape] = surfCoords_mm[shape][surfToLMorder[shape]]
    meanNorms_points[shape] = meanNorms_points[shape][surfToLMorder[shape]]
    # meanNorms_face[shape] = meanNorms_face[shape][newOrder]
    # faces[shape] = faces[shape][newOrder]

  # caseIndex = [i for i, pID in enumerate(patientIDs) if pID==case][0]

  #-reset new params
  landmarksAll=0
  transAll=0
  #-compile all lobar landmarks to one array
  for i, key in enumerate(subshapes):
    if i == 0: 
      origin["ALL"] = copy(origin[key])
      spacing["ALL"] = copy(spacing[key])
      drrArr["ALL"] = copy(drrArr[key])
      landmarks["ALL"] = lmProj[key].copy()
      lmTrans_total["ALL"]  = copy(lmTrans_total[key])
      meanNorms_points["ALL"] = copy(meanNorms_points[key])
      
    else: 
      landmarks["ALL"] = np.hstack((landmarks["ALL"], 
                                    lmProj[key])) 
      lmTrans_total["ALL"] = np.vstack((lmTrans_total["ALL"],lmTrans_total[key]))
      meanNorms_points["ALL"] = np.vstack((meanNorms_points["ALL"],
                                            meanNorms_points[key]))
    del origin[key], spacing[key], drrArr[key], \
      lmTrans_total[key], meanNorms_points[key], model[key], numModes[key]

  #-format data for testing by randomising selection and removing these 
  # from training
  assignedTestIDs = ["0645", "3948", "5268", "6730", "8865"]
  testSize = 5
  testID = []
  randomise_testing =  True
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
    testOrigin.append(origin["ALL"][t])
    testSpacing.append(spacing["ALL"][t])
    testIm.append(drrArr["ALL"][t])
    testLM.append(landmarks["ALL"][t])
    #-remove test data from train data
    patientIDs.pop(t)
    origin["ALL"] = np.delete(origin["ALL"],t,axis=0)
    spacing["ALL"] = np.delete(spacing["ALL"],t,axis=0)
    drrArr["ALL"] = np.delete(drrArr["ALL"],t,axis=0)
    landmarks["ALL"] = np.delete(landmarks["ALL"],t,axis=0)

  lmTrans_total["ALL"] = np.mean(lmTrans_total["ALL"], axis=0)
  #-create appearance model instance and load data
  ssam = SSAM(landmarks["ALL"], 
              landmarks["ALL"], 
              drrArr["ALL"], 
              origin["ALL"],
              spacing["ALL"],
              train_size=landmarks["ALL"].shape[0])
  density["ALL"]  = ssam.xg_train[:,:,-1]
  model["ALL"] = ssam.phi_sg
  meanArr = np.mean(landmarks["ALL"], axis=0)


  #-set number of modes
  numModes["ALL"] = np.where(
                      np.cumsum(ssam.pca_sg.explained_variance_ratio_)
                      >describedVariance)[0][0]
  print("modes used is", numModes)

  #-center the lobes vertically
  #-keep vertical alignment term for later use
  lmAlign = meanArr[:,2].mean() #landmarks["ALL"][:,2].mean()
  for key in surfCoords_mm.keys():
    surfCoords_mm[key][:,2] -= lmAlign
  meanArr[:,2] -= lmAlign


  surfCoordsBase = copy(surfCoords_mm)
  tagBase = copy(tag)
  for t, (tID, tImg, tOrig, tSpace) \
    in enumerate(zip(testID, testIm, testOrigin, testSpacing)):

    surfCoords_mm = surfCoordsBase.copy()
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
      # ax[1].scatter(landmarks["ALL"][caseIndex,:,0], landmarks["ALL"][caseIndex,:,2],s=1, c="black") 
      ax[1].scatter(edgePoints[:,0], edgePoints[:,1],s=2, c="blue") 
      plt.show()

    # exit()
    # if "ALL" in shapes:
    inputCoords = dict.fromkeys(["ALL"])
    inputCoords["ALL"] = meanArr.copy()
    #############################################################################
    #-declare posterior shape model class
    lpsm = LobarPSM(shape=inputCoords,#surfCoords_mm, 
                    xRay=edgePoints_mm,
                    normals=meanNorms_points, 
                    transform=lmTrans_total,
                    img=img,
                    # imgSpacing=spacing_xr,
                    imgCoords=imgCoords,
                    density=density,
                    model=model,
                    modeNum=numModes,
                    shapePairs=shapePairs,
                    c_edge=c_edge,
                    c_prior=c_prior,
                    c_dense=c_dense)
    #-import variables to class
    lpsm.variance = ssam.variance[:numModes["ALL"]]
    #-import functions to PSM class
    lpsm.getg_allModes = ssam.sam.getg_allModes
    lpsm.getDensity = ssam.sam.getDensity
    lpsm.normaliseTestImageDensity = ssam.sam.normaliseTestImageDensity
    # exit()
    #############################################################################

    lpsm.projLM, lpsm.projLM_ID = lpsm.getProjectionLandmarks(faces, 
                                                              meanNorms_face, 
                                                              surfCoords_mmOrig)

    lpsm.fissureLM_ID = 0 

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
    lobeBackup = copy(lpsm)
    if "ALL" in lpsm.projLM.keys():
      del lpms.projLM["ALL"], lpsm.projLM_ID["ALL"]

    lpsm.projLM, lpsm.projLM_ID = lpsm.deleteShadowedEdges(surfCoords_mm, 
                                                          lpsm.projLM, 
                                                          lpsm.projLM_ID,
                                                          )

    #-reorder projected points
    for key in subshapes:
      print("getting projlm for", key)
      if key=="RML": continue
      delInd = []
      for p, point in enumerate(lpsm.projLM_ID[key]):
        # projReorder.append(np.argwhere(point==surfToLMorder[key])[0][0])
        mappedLM = np.argwhere(point==surfToLMorder[key])
        if mappedLM.size>0:
          lpsm.projLM_ID[key][p] = mappedLM[0][0]
        else:
          delInd.append(p)
      #-delete projected surfPoints which were not included in mapping to LM space
      lpsm.projLM_ID[key] = np.delete(lpsm.projLM_ID[key], delInd)
      lpsm.projLM[key] = np.delete(lpsm.projLM[key], delInd)
    #-map projected landmark IDs for each lobe to their correpsonding position in
    #-the 'all' landmark array
    numLM = 0
    lpsm.projLM_ID["ALL"] = []
    for i, key in enumerate(subshapes):
      if i == 0:
         lpsm.projLM_ID["ALL"] = lpsm.projLM_ID[key].copy()
      else:
        if key != "RML": 
          lpsm.projLM_ID["ALL"] = np.append(lpsm.projLM_ID["ALL"],
                                          lpsm.projLM_ID[key]+numLM)
      numLM += len(surfCoords_mm[key])
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
    print(lpsm.projLM_ID["ALL"].max())
    # break

    lpsm.optimiseStage = "both" # tell class to optimise shape and pose
    lpsm.optIterSuc, lpsm.optIter = 0, 0
    lpsm.scale = optTrans_new["scale"]
    optAll = lpsm.optimisePoseAndShape(lpsm.objFunc, 
                                         initPose, 
                                         bounds, 
                                         epochs=numEpochs, 
                                         threads=1)
    print("\n\n\n\n\n\t\t\tTime taken is {0} ({1} mins)".format(time()-startTime,
                                                        round((time()-startTime)/60.),3))
    print(lpsm.projLM_ID["ALL"].max())
    #######################################################################
    

    density_plot = dict.fromkeys(inputCoords.keys())
    alignOutline = dict.fromkeys(inputCoords.keys())
    alignMean = dict.fromkeys(inputCoords.keys())
    morphedShape = dict.fromkeys(inputCoords.keys())
    alignMorphed = dict.fromkeys(inputCoords.keys())
    outlineMorphed = dict.fromkeys(inputCoords.keys())
    outShape = dict.fromkeys(inputCoords.keys())
    for counter in ["plot", "surf"]: #-plot overlay, then add lmAlign to save surf
      for key in inputCoords.keys():
        if counter == "surf":
          offset = np.array([0,0,lmAlign])
        else:
          offset = np.zeros(3)
        morphedShape[key] = lpsm.morphShape(inputCoords[key]+offset,
                                             lmTrans_total[key],
                                             optAll["b"][key],
                                             lpsm.model_s[key][:len(optAll["b"][key])]
                                             )
        alignMorphed[key], alignOutline[key] = getScaledAlignedLMs(morphedShape[key], 
                                                                    scale=optAll["scale"], 
                                                                    pose=optAll["pose"],
                                                                    transform=lmTrans_total[key],
                                                                    outlineIDs=lpsm.projLM_ID[key])
        # alignMorphed[key] = (lpsm.centerThenScale(morphedShape[key], 
        #                                         optAll["scale"],
        #                                         lmTrans_total[key])\
        #                       + np.insert(optAll["pose"], 1, 0))
        # # alignMorphed[key] = (lpsm.centerThenScale(morphedShape[key], 
        # #                                         optAll["scale"],
        # #                                         lmTrans_total[key])\
        # #                       + np.insert(optAll["pose"], 1, 0))
        # alignOutline[key] = alignMorphed[key][lpsm.projLM_ID[key]]
        outShape[key] = copy(alignMorphed[key])# + np.array([0,0, lmAlign])
        density_plot[key] = lpsm.getDensity(alignMorphed[key], img, imgCoords)
        print("DENSITY CHECK", np.mean(density_plot["ALL"]))
        match_g = np.argmin(np.mean(ssam.sam.density
                                    -density_plot["ALL"][np.newaxis,:], 
                                    axis=1)
                            )
        print("closest (gray-value) match is", match_g, "patient", patientIDs[match_g] )

        np.savetxt("results/outputCloud"+key+counter+tag+".txt", outShape[key])
      if counter == "plot":
        predictedSize = (alignMorphed["ALL"].max(axis=0)
                          -alignMorphed["ALL"].min(axis=0))
        # gtSize = (landmarks["ALL"][caseIndex].max(axis=0)
        #             -landmarks["ALL"][caseIndex].min(axis=0))
        # print("\npredicted size is", predictedSize)
        # print("landmark size is", gtSize)
        # print("size diff is ", gtSize-predictedSize,"\n")
        registered_output(alignMorphed, edgePoints_mm, img, optAll, tag="_all"+tag, 
                          density=density_plot)

        registered_output(alignOutline, edgePoints_mm, img, optAll, 
                          tag="_outline"+tag)
        #allLobeDensityError(lpsm.meanScaled, density_plot)
        allLobeDensityError(alignMorphed, density_plot, lpsm.meanScaled[key][:,-1],
                            tag=tag+"_errFromMean")

        
        # g_actual = ssam.sam.density[caseIndex]
        g_actual = ssam.sam.getDensity(testLM[t],img,imgCoords)
        allLobeDensityError(alignMorphed, density_plot, g_actual,
                            tag=tag+"_errFromXR")
        print("Mean error is", np.mean(abs(density_plot["ALL"]-g_actual)))

        sdist = np.sum( (morphedShape["ALL"]-inputCoords["ALL"])**2.,axis=1)**0.5
        # a = plt.scatter(morphedShape["ALL"][:,0], morphedShape["ALL"][:,2],c=sdist, s=1) ; plt.colorbar(a); plt.show()
        plt.close()
        a = plt.scatter(morphedShape["ALL"][:,0], morphedShape["ALL"][:,2],
                        cmap="seismic", 
                        c=sdist, s=1)
        cb = plt.colorbar(a)
        cb.set_label("distance [mm]", fontsize=11)
        plt.title("distance displaced from mean")
        plt.savefig("./images/xRayRecon/nevergrad/dist-change"+tag+".png")
        # plt.show()
      else:
        np.savetxt("./savedPointClouds/reconstructedLMs/"
                    +date_today+"_reconstructed"+tag+"_m_"+key+".dat",
                  outShape[key])
    # exit()
    # x=copy(surfCoords_mm)
    # y=copy(inputCoords)
    writeDict = dict.fromkeys(subshapes)
    #-map landmarks for all lobes to surface coordinates for specific lobes to write
    #-recenter shape
    inputCoords["ALL"][:,2] += lmAlign
    for key in subshapes:
      surfCoords_mm[key] = surfCoords[key]\
                            +np.loadtxt("savedPointClouds/"
                                                +"transformParams_m_"+key+".dat", 
                                                 skiprows=1, max_rows=1)
      print("\tmapping lobe", key)
      newOrder = []
      minList = []
      for point in surfCoords_mm[key]:
        newOrder.append(np.argmin(
                          utils.euclideanDist(inputCoords["ALL"], 
                                              point)))
        minList.append(np.min(
                          utils.euclideanDist(inputCoords["ALL"], 
                                              point)))
      print(np.min(minList), np.max(minList))
      writeDict[key] = outShape["ALL"][newOrder]

    lNums = {"RUL": "4",
               "RML": "5",
               "RLL": "6",
               "LUL": "7",
               "LLL": "8"
               }
    pointCounter = 0
    for key in writeDict.keys():
      print("\n{}".format(key))
      # print("\twriting lobe", key)
      # outCloud = v.densifyCloud(v.Points(writeDict[key]), 
      #                             1, closestN=5, maxIter=50, maxN=int(100e3))
      # print("\t num points:", outCloud.points().shape)
      # np.savetxt("./surfaces/"+date_today+"_reconstructed"+tag+"_m_"+key+".txt",
      #            outCloud.points())
      outCoords = outShape["ALL"][pointCounter:lmProj[key].shape[1]+pointCounter]
      pointCounter += lmProj[key].shape[1]
      # print("Checking outCoords shape", outCoords.shape[0])
      # print("lmProj[key].shape", lmProj[key].shape[1])
      # print("pointCounter", pointCounter, "vs total points", inputCoords["ALL"].shape[0])
      surfOutName = "./surfaces/"+date_today+"_reconstructed"+tag+"_m_"+key+".stl"

      lm_template = np.loadtxt(template_lm.format(key), delimiter=",")
      mesh_template = v.load(template_mesh.format(lNums[key]))

      morph = MorphTemplateMesh(lm_template, outCoords, mesh_template)
      v.write(morph.mesh_target, surfOutName)

      np.savetxt("./surfaces/"+date_today+"_NEWWRITEMETHODreconstructed"+tag+"_m_"+key+".txt",
                 outCoords)
      np.savetxt("./surfaces/"+date_today+"_reconstructed"+tag+"_m_"+key+".txt",
                 writeDict[key])
      # utils.writeAdjustedSTL(newPoints=writeDict[key], 
      #                      stlIn="./surfaces/totmean"+key+".stl",
      #                      outname="./surfaces/"+date_today\
      #                               +"_reconstructed"+tag+"_m_"+key+".stl")

    #-find closest matching item in training set
    dList = []
    mNorm = morphedShape["ALL"]-morphedShape["ALL"].mean()
    mNorm /= mNorm.std()
    for i in range(len(landmarks["ALL"])):
      dList.append(utils.euclideanDist(mNorm,
                                        ssam.x_vec_scale[i]).mean()
                  )
    match_s = np.argmin(dList)
    print("closest (shape) match is", match_s, "patient", patientIDs[match_s] )
    # exit()
''' 
plt.close()
fig, ax = plt.subplots(int(round(len(patientIDs)**0.5)), int(round(len(patientIDs)**0.5)))
ax = ax.ravel()
for i, pID in enumerate(patientIDs):
  ax[i].imshow(ssam.imgsN[i], cmap="gray")
  ax[i].set_xticks([])
  ax[i].set_yticks([])
plt.show()

plt.close()
for i, pID in enumerate(patientIDs):
  plt.imshow(ssam.imgsN[i], cmap="gray")
  plt.xticks([])
  plt.yticks([])
  plt.savefig("./images/xRayRecon/checkXRs/case"+pID+".png", format="png", dpi=300)

''' 