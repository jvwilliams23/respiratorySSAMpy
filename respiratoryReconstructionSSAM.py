"""
User args given by "python reconstructSSAM.py -h"

Creates Posterior Shape Model (PSM) of lung lobes based
on landmarks determined by GAMEs algorithm

This file only has class; no run script.

@author: Josh Williams

"""

import argparse
import random
from concurrent import futures
from copy import copy
from datetime import date
from distutils.util import strtobool
from glob import glob
from math import pi
from os import remove
from sys import argv, exit
from time import time

import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np
import vedo as v
from scipy.spatial.distance import cdist, pdist
from skimage import draw, filters, io
from skimage.color import rgb2gray
from skimage.filters import rank
from skimage.morphology import disk
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from vedo import printc

import userUtils as utils
from morphAirwayTemplateMesh import MorphAirwayTemplateMesh

# from reconstructSSAM import LobarPSM
from respiratorySAM import RespiratorySAM
from respiratorySSAM import RespiratorySSAM
from respiratorySSM import RespiratorySSM


class RespiratoryReconstructSSAM:
  def __init__(
    self,
    shape,
    xRay,
    lmOrder,
    normals,
    transform,
    density=None,
    img=None,
    imgCoords=None,
    imgCoords_all=None,
    imgCoords_axes=[0,2],
    model=None,
    modeNum=None,
    epochs=200,
    c_edge=1.0,
    c_prior=0.01,
    c_dense=0.5,
    c_anatomical=0.6,
    c_grad=0.4,
    kernel_distance=27,  # 18,9
    kernel_radius=16,
    quiet=False,
  ):  # 7):

    # tunable hyper-parameters
    self.c_edge = c_edge
    self.c_prior = c_prior
    self.c_dense = c_dense
    # to vary in parameter study!
    self.c_anatomical = c_anatomical
    self.c_grad = c_grad
    self.kernel_distance = kernel_distance
    self.kernel_radius = kernel_radius
    self.quiet = quiet

    self.lobes = ["RUL", "RML", "RLL", "LUL", "LLL"]

    # info from training dataset
    self.lmOrder = lmOrder
    self.shape = shape
    self.shapenorms = normals
    self.xRay = xRay  # -XR edge map
    self.projLM = None
    self.projLM_ID = None
    self.fissureLM_ID = None
    self.model = model
    self.transform = transform[np.newaxis]
    # -appearance model inputs
    self.density = density  # -landmark densities

    # number of images used for reconstruction (assumes img shape = 500 x 500)
    self.number_of_imgs = max(np.sum(img.shape)-1000,1)

    """
    # self.img = ssam.sam.normaliseTestImageDensity(img)
    # self.imgCoords = ssam.sam.drrArrToRealWorld(img,
    #                                             np.zeros(3), 
    #                                             spacing_xr)[0]
    # #-center image coords, so in the same coord system as edges
    # self.imgCoords -= np.array([250,250])*spacing_xr[[0,2]]
    """
    # format x-ray for image enhancement
    self.img = copy(img)  # -XR image array
    img = self.rescaleProjection(img)
    img *= 0.999  # avoid floating point error in scalar causing img > 1
    # img_filt = utils.bilateralfilter(img, 10)
    # if img_filt.max()>1:
    #   img_filt = img_filt/img_filt.max()
    img_local = utils.localNormalisation(img, 20)
    img_local = self.rescaleProjection(img_local)
    self.img_local = np.round(img_local, 4)
    print(self.img_local.min(), self.img_local.max())
    # self.imgGrad = rank.gradient(self.img_local, disk(5)) / 255
    # print(self.imgGrad.min(), self.imgGrad.max())
    self.imgCoords = imgCoords  # -X and Z coords of X-ray pixels
    self.imgCoords_all = imgCoords_all
    self.imgCoords_axes = imgCoords_axes
    self.alignTerm = xRay.mean(
      axis=0
    )  # -needed for coarse alignment coord frame

    self.optIter = 0
    self.optIterSuc = 0
    self.optStage = "align"

    self.scale = 1
    self.coordsAll = None
    self.projLM_IDAll = None
    self.fisIDAll = None

    self.optimiseStage = "pose"  # -first pose is aligned, then "both"
    # self.eng = 0

    if type(shape) == dict:
      # -initialise shape parameters for each sub-shape, to reduced mode nums
      self.model_s = dict.fromkeys(model.keys())
      self.model_g = dict.fromkeys(model.keys())
      self.b = np.zeros(modeNum)

      for k in model.keys():
        print(k)
        # get shape for reshaping 
        if self.density.ndim == 3:
          number_of_features = len(shape[k][0]) + len(self.density[0][0])
        else:
          number_of_features = len(shape[k][0]) + 1 # one value for density
        # -shape model components only
        self.model_s[k] = self.filterModelShapeOnly(self.model[k][:modeNum], number_of_features)
        # -gray-value model components only
        self.model_g[k] = self.filterModelDensityOnly(self.model[k][:modeNum], number_of_features)

      self.meanScaled = self.stackShapeAndDensity(
        self.scaleShape(shape["ALL"]), self.density.mean(axis=0)
      )
    else:
      # get shape for reshaping 
      if self.density.ndim == 3:
        number_of_features = len(shape[0]) + len(self.density[0])
      else:
        number_of_features = len(shape[0]) + 1 # one value for density
      # -parameters
      self.b = np.zeros(modeNum)
      # -shape model components only
      self.model_s = self.filterModelShapeOnly(self.model[:modeNum], number_of_features)
      # -gray-value model components only
      self.model_g = self.filterModelDensityOnly(self.model[:modeNum], number_of_features)
      self.meanScaled = self.stackShapeAndDensity(
        self.scaleShape(shape), self.density.mean(axis=0)
      )

  def rescaleProjection(self, img):
    """
    Set pixel value to [0:1] for a single or multiple projections.
    
    Parameter
    ---------
    image or series of images (np.ndarray, (500, 500), or (N, 500,500))

    Return
    ------
    Same array, bounded to a range of [0:1]
    """
    if img.ndim == 3:
      for i, img_proj in enumerate(img):
        scaler = MinMaxScaler()
        scaler.fit(img_proj)
        img[i] = scaler.transform(img_proj)
    elif img.ndim == 2:
      scaler = MinMaxScaler()
      scaler.fit(img)
      img = scaler.transform(img)
    return img

  def objFuncAirway(self, pose, scale=None, b=None):
    # -call initialised variables from __init__
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

    print("\nNext {0} loop iter {1}".format(self.optimiseStage, self.optIter))

    prior = []
    # -copy mean shape before shape is adjusted
    meanShape = copy(self.shape["ALL"])
    meanAirway = copy(self.shape["Airway"])
    # -call test parameters from optimizer
    self.b = copy(b)

    if not self.quiet:
      print("\t\t opt params ", pose, scale)
      print("\t\t\t", b)
    # -apply new shape parameters to each lobe
    all_morphed = self.morphAirway(
      meanShape,  # shape[key],
      meanShape.mean(axis=0),
      self.b,
      self.model_s["ALL"][: len(self.b)],
    )

    if pose.size == 2:
      pose = np.insert(pose, 1, 0)
    align = np.mean(all_morphed, axis=0)

    all_morphed = self.centerThenScale(all_morphed, scale, align)
    # -apply transformation to shape
    all_morphed = all_morphed + pose

    airway_morphed = all_morphed[self.lmOrder["Airway"]]
    # -check shape has not moved to be larger than XR or located outside XR
    # outside_bounds = np.any((all_morphed[:,2]>self.imgCoords[:,1].max())
    #                         | (all_morphed[:,2]<self.imgCoords[:,1].min())
    #                         | (all_morphed[:,0]>self.imgCoords[:,0].max())
    #                         | (all_morphed[:,0]<self.imgCoords[:,0].min())
    #                         )
    outside_bounds = np.any(
      (all_morphed[:, 2] < self.imgCoords_all[:, 2].min())
      | (all_morphed[:, 0] > self.imgCoords_all[:, 0].max())
      | (all_morphed[:, 0] < self.imgCoords_all[:, 0].min())
    )

    self.scale = scale  # -set globally to call in fitTerm
    # -intialise
    keyEdgeDists = 0
    fit = 0
    # -get losses
    lobe_morphed = dict.fromkeys(self.lobes)
    for lobe in self.lobes:
      lobe_morphed[lobe] = all_morphed[self.lmOrder[lobe]]
    fit = self.fitTerm(xRay, lobe_morphed, shapenorms)

    if self.number_of_imgs > 1:
      density_t = [None]*self.number_of_imgs
      for i in range(0, self.number_of_imgs):
        density_t[i] = self.getDensity(all_morphed, self.img[0], self.imgCoords_all[self.imgCoords_axes[i]])
      # convert list to array 
      # change from shape N_imgs, N_lms -> N_lms, N_imgs 
      density_t  = np.array(density_t).T
    else:
      density_t = self.getDensity(all_morphed, self.img, self.imgCoords).reshape(-1,1)

    print(all_morphed.shape, density_t.shape)
    # -TODO - TEST WITH modelled density instead of target?
    shapeIn = self.stackShapeAndDensity(self.scaleShape(all_morphed), density_t)
    # prior = np.sum(abs(b)/self.variance)
    prior = self.priorTerm(shapeIn, self.meanScaled)

    densityFit = self.densityLoss(
      density_t.reshape(-1),
      self.density.mean(axis=0).reshape(-1),
      self.model_g["ALL"][: len(self.b)],
      self.b,
    )
    print("\tfit loss {}\n\tdensity loss {}".format(fit, densityFit))
    print("\tprior loss", prior)  # round(prior,4))
    # self.c_edge = 0.2
    tallest_pt = airway_morphed[np.argmax(airway_morphed[:, 2])]
    # if self.c_grad != 0.0:
    #   gradFit = self.gradientTerm(airway_morphed, self.imgGrad, self.imgCoords)

    top_dist = 1.0 - np.exp(
      -1.0 * abs(tallest_pt[2] - self.imgCoords_all[:, 2].max()) / 5.0
    )
    # top_dist = abs(tallest_pt[2]-self.imgCoords[:,1].max())

    E = (
      (self.c_prior * prior) + (self.c_dense * densityFit) + (self.c_edge * fit)
    )
    E = (
      (self.c_edge * fit)
    )
    # if self.c_grad != 0.0:
    #   E += self.c_grad * gradFit
    # E += top_dist * 1.0
    # E += loss_anatomicalShadow
    if not self.quiet:
      print("top dist", top_dist)
      print('anatomical shadow and top dist off')
    if outside_bounds:
      if not self.quiet:
        print("OUTSIDE OF BOUNDS")
      # E += 0.25
      # return 2 # hard coded, assuming 2 is a large value for loss
    # if self.optIter > 1:
    loss_anatomicalShadow = self.c_anatomical * self.anatomicalShadow(
      self.img_local,
      self.imgCoords,
      airway_morphed,
      self.lmOrder,
      kernel_distance=self.kernel_distance,
      kernel_radius=self.kernel_radius,
    )
    print("\ttotal loss", E)

    if self.optIter % 250 == 0 and not self.quiet:
      if self.number_of_imgs == 1:
        self.overlayAirwayOnXR(self.img, all_morphed, scale, pose)
      elif self.number_of_imgs >= 2:
        self.overlayAirwayOnXR_multipleimgs(self.img, all_morphed, scale, pose)
      # exit()
    # if np.isnan(E):
    #   return 2
    # else:
    #   return E
    return E

  def gradientTerm(self, coords, imgGrad, imgCoords):
    """
    Inputs:
          coords (Nx3 np.ndarray):
          imgGrad (pixel x pixel, np.ndarray):
          imgCoords (pixel x 2 np.ndarray ):
    """
    lmGrad = self.getDensity(coords, imgGrad, imgCoords)[
      self.projLM_ID["Airway"]
    ]
    return (-1.0 * lmGrad).mean()

  def anatomicalShadow(
    self, img, img_coords, landmarks, lmOrder, kernel_distance, kernel_radius
  ):
    """
    anatomical shadow function proposed by
    Irving, B. et al., 2013. Proc. Fifth Int. Work. Pulm. Image Anal.

    Compares differences in a small kernel either side normal to a silhouette
    edge.

    loss = (c_in - c_out) / c_out
    where c is the average gray-value inside the kernel

    Inputs:
    img (numPixel x numPixel, np.ndarray): x-ray to get densities from.
    img_coords (numPixel x 2, np.ndarray): x-y coordinates for all pixels
    landmarks (N x 3, np.ndarray): landmark points on airways
    lmOrder (dict): IDs of landmarks for each shape (airway, skeleton, lobes)
    kernel_distance (int): number of pixels separating the inside and outside kernel
    kernel_radius (int): radius of kernel (units in pixels)

    returns:
    mean loss (float) for anatomical shadow
    """

    extent = [
      -self.img.shape[1] / 2.0 * self.spacing_xr[0],
      self.img.shape[1] / 2.0 * self.spacing_xr[0],
      -self.img.shape[0] / 2.0 * self.spacing_xr[2],
      self.img.shape[0] / 2.0 * self.spacing_xr[2],
    ]

    skeleton_ids = lmOrder["SKELETON"]
    airway_ids = lmOrder["Airway"]
    # get airway points not on skeleton (surface only)
    airway_surf_ids = airway_ids[~np.isin(airway_ids, skeleton_ids)]
    # get surface points that are in projected points list
    airway_surf_ids = airway_surf_ids[
      np.isin(airway_surf_ids, self.projLM_ID["Airway"])
    ]

    skel_pts = landmarks[skeleton_ids][:, [0, 2]]
    silhouette_pts = landmarks[airway_surf_ids][:, [0, 2]]

    dists = cdist(silhouette_pts, skel_pts)
    nearest_skel_pt = np.argmin(dists, axis=1)
    vec = silhouette_pts - skel_pts[nearest_skel_pt]
    div = np.sqrt(np.einsum("ij,ij->i", vec, vec))
    norm_vec = np.divide(vec, np.c_[div, div])

    all_p_in = (
      silhouette_pts + norm_vec * kernel_distance * self.spacing_xr[[0, 2]]
    )
    all_p_out = (
      silhouette_pts - norm_vec * kernel_distance * self.spacing_xr[[0, 2]]
    )
    # energy = np.zeros(len(all_p_out))
    energy = []
    delInd = []
    for p, (p_in, p_out) in enumerate(zip(all_p_in, all_p_out)):
      outside_bounds = np.any(
        (p_in[1] < img_coords[:, 1].min())
        | (p_in[1] > img_coords[:, 1].max())
        | (p_in[0] < img_coords[:, 0].min())
        | (p_in[0] > img_coords[:, 0].max())
        | (p_out[1] < img_coords[:, 1].min())
        | (p_out[1] > img_coords[:, 1].max())
        | (p_out[0] < img_coords[:, 0].min())
        | (p_out[0] > img_coords[:, 0].max())
      )
      if outside_bounds:
        delInd.append(p)
        continue
      # print(self.spacing_xr[[0,2]])
      # get nearest coord index
      p_in_index = [
        len(img) - 1 - np.argmin(abs(img_coords[:, 0] - p_in[0])),
        len(img) - 1 - np.argmin(abs(img_coords[:, 1] - p_in[1])),
      ]
      p_out_index = [
        len(img) - 1 - np.argmin(abs(img_coords[:, 0] - p_out[0])),
        len(img) - 1 - np.argmin(abs(img_coords[:, 1] - p_out[1])),
      ]
      # print(p_in, p_out)
      # print(p_in_index, p_out_index)
      # print(norm_vec[p])
      # get anatomical shadow value
      # c_in = img[draw.circle(p_in_index[1], p_in_index[0],
      #                         kernel_radius, img.shape)]
      # c_out = img[draw.circle(p_out_index[1], p_out_index[0],
      #                         kernel_radius, img.shape)]
      c_in = img[
        draw.circle(p_in_index[1], p_in_index[0], kernel_radius, img.shape)
      ]
      c_out = img[
        draw.circle(p_out_index[1], p_out_index[0], kernel_radius, img.shape)
      ]

      # if self.optIter % 100:
      #   img_in = img.copy()
      #   img_in[draw.circle(p_in_index[1], p_in_index[0],
      #                           kernel_radius, img.shape)] = 0

      #   img_out = img.copy()
      #   img_in[draw.circle(p_out_index[1], p_out_index[0],
      #                           kernel_radius, img.shape)] = 1
      #   plt.close()
      #   airway_all_pts = landmarks[airway_ids][:,[0,2]]
      #   fig, ax =  plt.subplots(1,2)
      #   ax.ravel()
      #   ax[0].imshow(img, cmap='gray', extent=extent)
      #   ax[0].scatter(airway_all_pts[:,0], airway_all_pts[:,1],s=2,c='black')
      #   ax[1].imshow(img_in, cmap='gray', extent=extent)
      #   # ax[1].imshow(img_out, cmap='gray', extent=extent)
      #   ax[1].scatter(silhouette_pts[p,0], silhouette_pts[p,1], s=2, c='blue')
      #   ax[1].scatter(skel_pts[:,0], skel_pts[:,1],s=2,c='black')
      #   plt.savefig('images/reconstruction/debug/shadow{}.png'.format(self.optIter))

      energy_at_p = (c_in.mean() - c_out.mean()) / c_out.mean()
      if not np.isnan(energy_at_p):
        energy.append(energy_at_p)
      else:
        delInd.append(p)
    # print(energy)
    energy = np.array(energy)
    silhouette_pts = np.delete(silhouette_pts, delInd, axis=0)

    # if self.optIter % 500 == 0 and not self.quiet:
    #   plt.close()
    #   # for debugging anatomical shadow values
    #   fig, ax = plt.subplots()
    #   ax.imshow(img, cmap='gray', extent=extent)
    #   scatter = ax.scatter(silhouette_pts[:,0], silhouette_pts[:,1],
    #                         c=energy, s=2)
    #   plt.colorbar(scatter)
    #   plt.savefig('images/reconstruction/debug/iter{}shadow.png'.format(self.optIter))
    # '''
    # # plt.show()
    # # exit()
    # '''

    # print(energy)
    # account for empty arrays when all points are outside of the domain
    if len(energy) == 0:
      return 0
    else:
      print("\tanatomicalShadow", energy.sum(), energy.mean())
      return (energy).mean()

  def scaleShape(self, shape):
    """
    return shape (lm x 3 array) with 0 mean and 1 std
    """
    return (shape - shape.mean(axis=0)) / shape.std(axis=0)
    # return (shape-shape.mean(axis=0))/shape.var(axis=0)

  def centerThenScale(self, shape, scale, alignTerm):
    """
    Center shape and then increase by isotropic scaling.
    Removes effect of offset on scaling.
    """
    shape = shape - alignTerm
    shape = shape * scale
    return shape + alignTerm

  def stackShapeAndDensity(self, shape, density):
    """
    Inputs:
            shape: array (lm x 3)
            density: array (lm x nproj)
    Outputs: array(lm x 3+nproj)
    """
    return np.hstack((shape, density))

  def filterModelShapeOnly(self, model, number_of_features=4):
    """
    Return model without density in columns.
    Input 2D array, shape = ( nFeature, 4n )
    Return 2D array, shape = ( nFeature, 3n )
    where n = num landmarks
    """
    # no appearance params
    model_as_columns = model.reshape(model.shape[0], -1, number_of_features)
    number_of_appearances_cols = number_of_features - 3
    # slice to remove columns representing appearance/density
    model_noApp = model_as_columns[:,:,:-number_of_appearances_cols]
    print("model_noApp.shape", model_noApp.shape)
    # -reshape to 2D array
    return model_noApp.reshape(model.shape[0], -1)

  def filterModelDensityOnly(self, model, number_of_features=4):
    """
    Return model without shape in columns.
    Input 2D array, shape = ( nFeature, 4n )
    Return 2D array, shape = ( nFeature, n )
    where n = num landmarks
    """
    # no appearance params
    model_as_columns = model.reshape(model.shape[0], -1, number_of_features)
    number_of_appearances_cols = number_of_features - 3
    # slice to remove columns representing appearance/density
    model_no_shape = model_as_columns[:,:,-number_of_appearances_cols:]
    # -reshape to 2D array
    print("model_no_shape.shape", model_no_shape.shape)
    return model_no_shape.reshape(model.shape[0], -1)

  def normaliseDist(self, dist):
    """
    Normalise a distance or list of distances to have range [0,1]
    """
    return np.exp(-1 * np.array(dist) / 5)

  def densityLoss(self, density_t, densityMean, model, b):
    """ """
    # -modelled density
    density_m = self.getg_allModes(
      densityMean, model, b * np.sqrt(self.variance)
    )
    # -target density (i.e. density at the nearest pixel)
    # density_t = density_t #self.getDensity(lm, img, imgCoords)
    abs_diff = abs(density_t - density_m)
    return abs_diff.max()
    # return np.mean(abs_diff)

  def morphShape(self, shape, transform, shapeParams, model):
    """
    Adjust shape transformation and scale.
    Imports to SSM and extracts adjusted shape.
    """
    removeTrans = shape - transform
    removeMean = removeTrans.mean(axis=0)
    shapeCentre = removeTrans - removeMean
    scaler = shapeCentre.std(axis=0)
    shapeSc = (
      shapeCentre / scaler
    )  # StandardScaler().fit_transform(shapeCentre)

    # shapeOut = shapeSc + np.dot(model.T, #-04/11/20 - test diff normalisation
    #                             shapeParams).reshape(-1,3)
    shapeOut = shapeSc + np.dot(
      shapeParams[np.newaxis, :] * np.sqrt(self.variance[np.newaxis, :]), model
    ).reshape(-1, 3)

    shapeDiff = np.sqrt(np.sum((shapeOut - shapeSc) ** 2, axis=1))
    if not self.quiet:
      print(
        "shape diff [normalised] \t mean",
        np.mean(shapeDiff),
        "\t max",
        np.max(shapeDiff),
      )
    # shapeOut = ssam.getx_allModes(shapeSc.reshape(-1),
    #                                 model,
    #                                 shapeParams)

    # shapeOut = ((shapeOut*shapeSc.std(axis=0)
    #             +shapeSc.mean(axis=0))
    #             *scaler) \
    #             +removeMean+transform
    shapeOut = (shapeOut * scaler) + removeMean + transform

    shapeDiff = np.sqrt(np.sum((shapeOut - shape) ** 2, axis=1))
    if not self.quiet:
      print(
        "shape diff [real space] \t mean",
        np.mean(shapeDiff),
        "\t max",
        np.max(shapeDiff),
      )

    return shapeOut

  def morphAirway(self, shape, transform, shapeParams, model):
    """
    Adjust shape transformation and scale.
    Imports to SSM and extracts adjusted shape.
    """
    # removeTrans = shape - transform
    removeMean = shape.mean(axis=0)
    shapeCentre = shape - removeMean
    scaler = shapeCentre.std(axis=0)
    shapeSc = (
      shapeCentre / scaler
    )  # StandardScaler().fit_transform(shapeCentre)

    # shapeOut = shapeSc + np.dot(model.T, #-04/11/20 - test diff normalisation
    #                             shapeParams).reshape(-1,3)
    shapeOut = shapeSc + np.dot(
      shapeParams[np.newaxis, :] * np.sqrt(self.variance[np.newaxis, :]), model
    ).reshape(-1, 3)

    shapeDiff = np.sqrt(np.sum((shapeOut - shapeSc) ** 2, axis=1))
    if not self.quiet:
      print(
        "shape diff [normalised] \t mean",
        np.mean(shapeDiff),
        "\t max",
        np.max(shapeDiff),
      )
    # shapeOut = ssam.getx_allModes(shapeSc.reshape(-1),
    #                                 model,
    #                                 shapeParams)

    # shapeOut = ((shapeOut*shapeSc.std(axis=0)
    #             +shapeSc.mean(axis=0))
    #             *scaler) \
    #             +removeMean+transform
    shapeOut = (shapeOut * scaler) + removeMean

    shapeDiff = np.sqrt(np.sum((shapeOut - shape) ** 2, axis=1))
    if not self.quiet:
      print(
        "shape diff [real space] \t mean",
        np.mean(shapeDiff),
        "\t max",
        np.max(shapeDiff),
      )

    return shapeOut

  def optimiseAirwayPoseAndShape(
    self, objective, init, bounds, epochs=2, threads=1
  ):
    """
    Minimises objective function using Nevergrad gradient-free optimiser
    """
    instrum = ng.p.Instrumentation(
      pose=ng.p.Array(init=init[:2]).set_bounds(  # shape=(3,),
        bounds[:2, 0], bounds[:2, 1]
      ),
      scale=ng.p.Scalar(init=init[2]).set_bounds(  # Scalar(
        bounds[2, 0], bounds[2, 1]
      ),
      b=ng.p.Array(init=np.zeros(self.b.size)).set_bounds(
        bounds[3, 0], bounds[3, 1]
      ),
    )

    optimizer = ng.optimizers.NGO(  # CMA(#NGO(
      parametrization=instrum, budget=epochs, num_workers=threads
    )

    if threads > 1:
      with futures.ThreadPoolExecutor(
        max_workers=optimizer.num_workers
      ) as executor:
        recommendation = optimizer.minimize(
          objective, executor=executor, batch_mode=True
        )
    else:
      # recommendation = optimizer.minimize(objective)
      lossLog = []
      recommendation = optimizer.provide_recommendation()
      for _ in range(optimizer.budget):
        x = optimizer.ask()
        loss = objective(*x.args, **x.kwargs)
        optimizer.tell(x, loss)
        lossLog.append(loss)
    recommendation = (
      optimizer.provide_recommendation()
    )  # -update recommendation

    tag = ""
    utils.plotLoss(
      lossLog, stage=self.optimiseStage, wdir="images/reconstruction/"
    )  # -plot loss

    optOut = dict.fromkeys(["pose", "scale", "b"])
    # optOut = dict.fromkeys(["pose","b"])
    optOut["pose"] = recommendation.value[1]["pose"]
    optOut["scale"] = recommendation.value[1]["scale"]
    optOut["b"] = recommendation.value[1]["b"]
    print("recommendation is", recommendation.value)

    return optOut

  def fitTerm(self, xRay, shapeDict, pointNorms3DDict):

    v = 5.0  # distance weighting factor
    n = 0  # initialise number of points
    thetaList = []  # * len(shapeDict.keys())

    # plt.close()
    # plt.plot(xRay[:,0], xRay[:,1], lw=0, marker="o", ms=2, c="black")

    for k, key in enumerate(self.lobes):
      # -get only fd term for RML
      if key != "RML":

        shape = shapeDict[key][self.projLM_ID[key]][:, [0, 2]]
        # plt.scatter(shape[:,0], shape[:,1])
        # pointNorms3D = pointNorms3DDict[key][self.projLM_ID[key]]
        d_i = np.zeros(shape.shape[0])
        n += len(shape)
        if len(xRay.shape) > 2:  # -if xRay is a 3D array
          n_proj = xRay.shape[2]
        else:
          n_proj = 1
        theta = np.zeros(shape.shape[0])

        # -get distance term (D_i)
        distArr = cdist(shape, xRay)
        d_i = np.min(distArr, axis=1)
        D_i = np.exp(-d_i / v)
        theta = abs(1 - D_i)

        thetaList.append(np.sum(theta))
    E_fit = (1 / (n * n_proj)) * np.sum(thetaList)

    return E_fit

  def priorTerm(self, shape, meanShape):
    """
    Compare shape generated by optimisation of shape parameters
    with mean shape of dataset, using mahalanobis distance.
    """

    # -centre shape
    # meanShapeAligned = meanShape-np.mean(meanShape, axis=0)
    # shapeAligned = shapeDict-np.mean(shapeDict, axis=0)

    # -get avg Mahalanobis dist from mean shape
    # E_prior = np.mean(utils.mahalanobisDist(meanShape,
    #                                         shape)
    #                     )
    E_prior = (
      np.sum(utils.mahalanobisDist(meanShape, shape)) / meanShape.shape[0]
    )

    return E_prior

  def overlayAirwayOnXR(self, img, coords, scale, pos, tag=""):
    extent = [
      -self.img.shape[1] / 2.0 * self.spacing_xr[0],
      self.img.shape[1] / 2.0 * self.spacing_xr[0],
      -self.img.shape[0] / 2.0 * self.spacing_xr[2],
      self.img.shape[0] / 2.0 * self.spacing_xr[2],
    ]
    plt.close()
    plt.imshow(img, cmap="gray", extent=extent)
    plt.scatter(
      self.xRay[:, 0], self.xRay[:, 1], s=4, c="black"
    )  # , alpha=0.2)
    # plt.scatter(coords[self.projLM_IDAll,0],
    #             coords[self.projLM_IDAll,2],
    #             s=2, c='yellow')
    for key in self.projLM_ID.keys():
      # if key == 'RML':
      #   continue
      # else:
      projLM_key = self.projLM_ID[key]
      # print(key, projLM_key)
      plt.scatter(
        coords[self.lmOrder[key]][projLM_key, 0],
        coords[self.lmOrder[key]][projLM_key, 2],
        s=8,
        c="yellow",
      )

    # plt.text(self.imgCoords[:,0].min()*0.9,self.imgCoords[:,1].max()*0.9,
    #          "pos {}, scale{}".format(str(pos), str(scale)))
    # plt.savefig(
    #         'images/reconstruction/debug/iter{}{}.png'.format(str(self.optIter),
    #                                                           tag)
    #             )
    # formatting to remove whitespace!
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(
      "images/reconstruction/debug/iter{}{}.png".format(str(self.optIter), tag),
      bbox_inches="tight",
      pad_inches=0,
    )
    # exit()
    return None

  def overlayAirwayOnXR_multipleimgs(self, img, coords, scale, pos, tag=""):
    for i, im_i in enumerate(img):
      extent = [
        -im_i.shape[1] / 2.0 * self.spacing_xr[self.imgCoords_axes[i][0]],
        im_i.shape[1] / 2.0 * self.spacing_xr[self.imgCoords_axes[i][0]],
        -im_i.shape[0] / 2.0 * self.spacing_xr[self.imgCoords_axes[i][1]],
        im_i.shape[0] / 2.0 * self.spacing_xr[self.imgCoords_axes[i][1]],
      ]
      plt.close()
      plt.imshow(im_i, cmap="gray", extent=extent)
      # plt.scatter(
      #   self.xRay[:, 0], self.xRay[:, 1], s=4, c="black"
      # ) 
      for key in self.projLM_ID.keys():
        # if key == 'RML':
        #   continue
        # else:
        projLM_key = self.projLM_ID[key]
        # print(key, projLM_key)
        plt.scatter(
          # coords[self.lmOrder[key]][projLM_key, 0],
          # coords[self.lmOrder[key]][projLM_key, 2],
          coords[self.lmOrder[key]][:, self.imgCoords_axes[i][0]],
          coords[self.lmOrder[key]][:, self.imgCoords_axes[i][1]],
          s=8,
          c="yellow",
        )
      # formatting to remove whitespace!
      plt.gca().set_axis_off()
      plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
      plt.margins(0, 0)
      plt.gca().xaxis.set_major_locator(plt.NullLocator())
      plt.gca().yaxis.set_major_locator(plt.NullLocator())
      plt.savefig(
        "images/reconstruction/debug/img{}-iter{}{}.png".format(i, str(self.optIter), tag),
        bbox_inches="tight",
        pad_inches=0,
      )
    # exit()
    return None

  def getProjectionLandmarks(self, faceIDs, faceNorms, points):
    """
    args:
        faceIDs array(num faces, 3): Each row has three IDs corresponding
                to the vertices that construct that face.
        faceNorms array(num faces, 3): components of each face normal vector
        points array(num points, 3): coordinates of each surface point
    returns:
        projectionLM array(n, 2): 2D projection coordinates of silhouette landmarks
    """
    assert (
      type(points) == type(faceIDs) == type(faceNorms)
    ), "type mismatch in surface faces, normals and points"

    if type(points) == dict:
      projectionLM = dict.fromkeys(points.keys())
      projectionLM_ID = dict.fromkeys(points.keys())

      for shape in points.keys():
        norms = []
        projectionLM[shape] = []
        projectionLM_ID[shape] = []
        for pID in range(len(points[shape])):
          norms.append(
            np.where(
              (faceIDs[shape][:, 0] == pID)
              | (faceIDs[shape][:, 1] == pID)
              | (faceIDs[shape][:, 2] == pID)
            )[0]
          )
          if len(norms[pID]) > 1:
            """check if y normal for point has +ve and -ve components
            in the projection plane"""
            if (
              np.min(faceNorms[shape][norms[pID]][:, 1]) < 0
              and np.max(faceNorms[shape][norms[pID]][:, 1]) > 0
            ):
              projectionLM[shape].append(points[shape][pID])
              projectionLM_ID[shape].append(pID)
            # else:
            #   print(np.array(faceNorms[shape][norms[pID]][:,1]),
            #         len(faceNorms[shape][norms[pID]][:,1]))

        projectionLM[shape] = np.array(projectionLM[shape])
        # -delete projection plane from coords
        projectionLM[shape] = np.delete(projectionLM[shape], 1, axis=1)
        projectionLM_ID[shape] = np.array(projectionLM_ID[shape])
    else:
      norms = []
      projectionLM = []
      projectionLM_ID = []
      for pID in range(len(points)):
        norms.append(
          np.where(
            (faceIDs[:, 0] == pID)
            | (faceIDs[:, 1] == pID)
            | (faceIDs[:, 2] == pID)
          )[0]
        )
        if len(norms[pID]) > 1:
          """check if y normal for point has +ve and -ve components
          in the projection plane"""
          if (
            np.min(faceNorms[norms[pID]][:, 1]) < 0
            and np.max(faceNorms[norms[pID]][:, 1]) > 0
          ):
            projectionLM.append(points[pID])
            projectionLM_ID.append(pID)
        else:
          continue
      ids = np.arange(0, len(points))
      np.where(
        np.isin(faceIDs[:, 0], points)
        | (faceIDs[:, 1] == pID)
        | (faceIDs[:, 2] == pID)
      )[0]

      """
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
      """
      projectionLM = np.array(projectionLM)
      # -delete projection plane from coords
      projectionLM = np.delete(projectionLM, 1, axis=1)
      projectionLM_ID[shape] = np.array(projectionLM_ID[shape])
    return projectionLM, projectionLM_ID

  def saveSurfProjectionComparison(self, E, xRay):
    print("\n\tFIT IS", E)

    plt.text(
      xRay[:, 0].min() * 0.9,
      xRay[:, 1].max() * 0.9,
      "E = {0}".format(round(E, 6)),
    )
    plt.savefig(
      "images/xRayRecon/nevergrad/"
      + self.optimiseStage
      + str(self.optIter)
      + ".png"
    )
    return None

  def deleteShadowedEdges(self, coords, projLM, projLM_ID):
    """
    Deletes landmarks that are not found on the radiograph
    (overlapped by spine or fissures that are not visible)
    """
    # if "RUL" in coords.keys()\
    # and "RLL" in coords.keys():
    #   lpsm.shape = { filterKey: coords[filterKey] \
    #               for filterKey in ["RUL","RLL"] }
    # else:
    shape = copy(coords)  # -why?

    widthMaxCutoff = 0.5
    if "RUL" in projLM.keys():
      upperKey = "RUL"
      heightMax = 0.8
    else:
      upperKey = "RLL"
      if "RLL" not in projLM.keys():
        return projLM, projLM_ID
      heightMax = 1.3
    height = projLM[upperKey][:, 1].max() - projLM["RLL"][:, 1].min()
    for key in coords.keys():
      print(key)
      # if key[0] == "L":
      #   continue
      delInd = []
      if key == "RML" or key == "Airway":
        projLM_ID[key] = np.array(projLM_ID[key])
        continue
      if key == "RLL":
        heightMin = -0.01
        widthCutoff = 0.65
        widthMaxCutoff = 0.5  # 2
      else:  # if key == "RLL":
        heightMin = -0.01  # 0.3
        widthCutoff = 0.85
        widthMaxCutoff = 1
      width = projLM[key][:, 0].max() - projLM[key][:, 0].min()
      if key != "LLL":
        delInd = np.where(
          (projLM[key][:, 0] - projLM[key][:, 0].min() > widthCutoff * width)
          | (
            (projLM[key][:, 1] - projLM["RLL"][:, 1].min() > heightMin * height)
            & (
              projLM[key][:, 1] - projLM["RLL"][:, 1].min() < heightMax * height
            )
            & (projLM[key][:, 0] - projLM[key][:, 0].min() > 0.2 * width)
            & (
              projLM[key][:, 0] - projLM[key][:, 0].min()
              < widthMaxCutoff * width
            )  # JW 26/06/20
          )
        )[0]
      if key == "RUL":
        # -filter low points
        # -(corresponds to curvature at RUL RLL intersection)
        delInd = np.unique(
          np.append(
            delInd,
            np.where(
              projLM[key][:, 1]
              < ((projLM[key][:, 1].max() - projLM[key][:, 1].min()) * 0.2)
              + projLM[key][:, 1].min()
            )[0],
          )
        )

        projLM[key] = np.delete(projLM[key], delInd, axis=0)
        projLM_ID[key] = np.delete(projLM_ID[key], delInd)
        del delInd  # -delete to avoid accidental deletion of wrong indexes

        """filter outlier clusters by applying 75th percentile 
            mahalanobis distance filter twice"""
        tmpprojLM = copy(projLM[key])  # -store initial values
        for i in range(2):
          # -keep data to outer side of RUL
          dataKept = projLM["RUL"][
            np.where(
              projLM["RUL"][:, 0]
              < (
                projLM["RUL"][:, 0].min()
                + (projLM["RUL"][:, 0].max() - projLM["RUL"][:, 0].min()) / 2
              )
            )
          ]
          # -set aside inner coordinates for filtering
          data = projLM["RUL"][
            np.where(
              projLM["RUL"][:, 0]
              > (
                projLM["RUL"][:, 0].min()
                + (projLM["RUL"][:, 0].max() - projLM["RUL"][:, 0].min()) / 2
              )
            )
          ]

          md = cdist(data, data, "mahalanobis")
          # -set coords with high mean mahalanobis distance for deletion
          delInd = np.unique(
            np.where(md.mean(axis=0) > np.percentile(md.mean(axis=0), 65))[0]
          )

          data = np.delete(data, delInd, axis=0)
          del delInd  # -delete to avoid accidental deletion of wrong indexes
          projLM[key] = np.vstack((dataKept, data))  # -reset array
        # -loop to find landmark ID's removed
        idkept = []
        for lmID in projLM_ID["RUL"]:
          if tmpprojLM[np.where(projLM_ID["RUL"] == lmID)] in projLM["RUL"]:
            idkept.append(lmID)
        projLM_ID["RUL"] = np.array(idkept)
      elif key == "RLL":
        """
        delete upper half of landmark points (due to fissure overlap)
        OR
        inner fifth (0.2) of landmarks (not visible on X-ray)
        """
        delInd = np.where(
          (
            (projLM["RLL"][:, 1] - projLM["RLL"][:, 1].min() > 0.6 * height)
            | ((projLM["RLL"][:, 0] - projLM["RLL"][:, 0].min() > 0.8 * width))
          )
        )

        # -delete previous stored indexes
        projLM[key] = np.delete(projLM[key], delInd, axis=0)
        projLM_ID[key] = np.delete(projLM_ID[key], delInd)

        # -filter out upper right points in the RLL
        # -set coords with high mean mahalanobis distance for deletion
        delInd = np.unique(
          np.where(md.mean(axis=0) > np.percentile(md.mean(axis=0), 92.5))[0]
        )
        projLM[key] = np.delete(projLM[key], delInd, axis=0)
        projLM_ID[key] = np.delete(projLM_ID[key], delInd)

        for i in range(2):
          # -find lowest point in RLL to filter out
          # -this is needed as it is typically not segmented by XR seg tool
          cornerP = projLM["RLL"][np.argmin(projLM["RLL"][:, 1], axis=0)]
          cutoff = 0.4  # -define cutoff distance

          # -use mahalanobis distance to filter out points
          projLM_ID["RLL"] = projLM_ID["RLL"][
            np.where(utils.mahalanobisDist(projLM["RLL"], cornerP) > cutoff)
          ]
          projLM["RLL"] = projLM["RLL"][
            np.where(utils.mahalanobisDist(projLM["RLL"], cornerP) > cutoff)
          ]
      elif key == "LUL":  # -if left lung
        delInd = np.where(
          (
            (projLM[key][:, 0] - projLM[key][:, 0].min() < 0.4 * width)
            & (projLM[key][:, 1] - projLM[key][:, 1].min() < 0.8 * height)
          )
          | (projLM[key][:, 1] - projLM[key][:, 1].min() < 0.4 * height)
        )
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
  parser = argparse.ArgumentParser(description="SSM for lung lobe variation")
  parser.add_argument(
    "--inp",
    "-i",
    default=False,
    type=str,
    required=True,
    help="input files (landmarks)",
  )
  parser.add_argument(
    "--case",
    "-c",
    default=False,
    type=str,  # , required=True,
    help="training data case",
  )
  parser.add_argument(
    "--out",
    "-o",
    default=False,
    type=str,
    required=True,
    help="output surface tag ",
  )
  parser.add_argument(
    "--randomise",
    "-r",
    default=str(True),
    type=strtobool,
    help="randomise testing set? [default = true]",
  )
  parser.add_argument(
    "--var",
    "-v",
    default=0.7,
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
    "--c_dense",
    "-cd",
    default=0.25,
    type=float,
    help="density loss coefficient",
  )
  parser.add_argument(
    "--c_edge", "-ce", default=1.0, type=float, help="edge map loss coefficient"
  )
  parser.add_argument(
    "--drrs", default=False, type=str, required=True, help="input files (drr)"
  )
  parser.add_argument(
    "--meshdir",
    "-m",
    default=False,
    type=str,
    required=True,
    help="directory of surface files",
  )
  parser.add_argument(
    "--shapes",
    default="*",
    type=str,
    help="which shape would the user like to grow?"
    + "Corresponds to string common in landmarks text files"
    + "\nRUL, RML, RLL, LUL, LLL, or ALL?",
  )
  parser.add_argument(
    "--debug",
    default=False,
    type=bool,
    help="debug mode -- shows print checks and blocks" + "plotting outputs",
  )
  parser.add_argument(
    "--epochs",
    "-e",
    default=500,
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
    "--imgSpacing",
    default=1,
    type=int,
    help="multiplier to coarsen images (must be int)",
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

  return (
    inputDir,
    case,
    tag,
    var,
    drrDir,
    debugMode,
    shapeKey,
    surfDir,
    numEpochs,
    xray,
    c_edge,
    c_dense,
    c_prior,
    imgSpacing,
  )


def getScaledAlignedLMs(coords, scale, transform, pose, outlineIDs):
  scaled = lpsm.centerThenScale(coords, scale, transform) + np.insert(
    pose, 1, 0
  )
  return scaled, scaled[outlineIDs]


def allLobeDensityError(meanScaled, densityOut, densityMean=0, tag=""):
  """
  Plots the error in gray-value of reconstruction compared to the mean
  """
  plt.close()
  fig, ax = plt.subplots(
    nrows=1, ncols=len(meanScaled.keys()), figsize=(16 / 2.54, 10 / 2.54)
  )

  for i, key in enumerate(meanScaled.keys()):
    if len(meanScaled.keys()) == 1:
      ax_set = ax
    else:
      ax_set = ax[i]
    xbar = meanScaled[key][:, [0, 1, 2]]
    gbar = densityMean  # meanScaled[key][:,-1]
    gout = densityOut[key]
    a = ax_set.scatter(
      xbar[:, 0],
      xbar[:, 2],
      cmap="seismic",
      c=abs(gbar - gout).reshape(-1),
      vmin=0,
      vmax=1,
      s=1,
    )
    ax_set.axes.xaxis.set_ticks([])
    ax_set.axes.yaxis.set_ticks([])
    ax_set.set_title(str(key), fontsize=12)
  # -set colorbar
  if len(meanScaled) == 1:
    cb = fig.colorbar(a, ax=ax)
  else:
    cb = fig.colorbar(a, ax=ax.ravel().tolist())
  cb.set_label("density", fontsize=11)
  fig.suptitle("Density error in reconstruction", fontsize=12)
  # plt.show()
  fig.savefig(
    "./images/xRayRecon/nevergrad/density-error" + tag + ".png",
    pad_inches=0,
    format="png",
    dpi=300,
  )
  return None


def visualiseOutput(coords, scale=[1, 1, 1], pose=[0, 0, 0]):
  # visualiseOutput(surfCoords_mm, optTrans["scale"], optTrans["pose"])
  v = []
  for key in coords.keys():
    v.append(v.Points((coords[key] * scale[key]) + pose[key], r=4))
    np.savetxt(
      key + "coords.txt",
      (coords[key] * scale[key]) + pose[key],
      delimiter=",",
      header="x,y,z",
    )
  v.show(v[0], v[1], v[2])
  return None


def loadXR(file):
  """

  TODO MAYBE NEED TO NORMALISE???????????????????
  IT IS NORMALISED IN USERUTILS.PY
  """
  g_im = rgb2gray(io.imread(file))
  g_im[0] = g_im[1].copy()
  g_im = utils.he(g_im[::imgSpaceCoeff, ::imgSpaceCoeff])
  return g_im


def registered_output(coords, outline, img, optStage, tag="", density=None):
  plt.close()
  plt.imshow(img, extent=extent, cmap="gray")
  plt.scatter(outline[:, 0], outline[:, 1], s=1, color="black")

  for key in coords.keys():
    if density:
      plt.scatter(
        coords[key][:, 0],
        coords[key][:, 2],
        s=1,
        c=density[key],
        cmap="gray",
        vmin=-1,
        vmax=1,
      )
    else:
      plt.scatter(coords[key][:, 0], coords[key][:, 2], s=1, c="yellow")
  plt.text(
    edgePoints_mm[:, 0].min() * 0.9,
    edgePoints_mm[:, 1].max() * 1.2,
    "trans " + str(optStage["pose"]) + "\nscale " + str(optStage["scale"]),
  )
  plt.savefig(
    "images/xRayRecon/nevergrad/" + lpsm.optimiseStage + "Final" + tag + ".png"
  )
  return None


if __name__ == "__main__":
  date_today = str(date.today())
  print(__doc__)
