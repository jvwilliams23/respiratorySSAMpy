"""
User args given by "python reconstructSSAM.py -h"

Creates Posterior Shape Model (PSM) of lung lobes based
on landmarks determined by GAMEs algorithm

This file only has class; no run script.

@author: Josh Williams

"""

import argparse
from copy import copy

import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np
from scipy.spatial.distance import cdist
from skimage import draw
from sklearn.preprocessing import MinMaxScaler

import userUtils as utils

from respiratorySSAM import RespiratorySSAM
# import pyssam


class RespiratoryReconstructSSAM(RespiratorySSAM):
  def __init__(
    self,
    ssam_obj,
    shape,
    xRay,
    lmOrder,
    normals,
    transform,
    density=None,
    img=None,
    imgCoords=None,
    imgCoords_axes=[0, 2],
    model=None,
    modeNum=None,
    epochs=200,
    c_edge=0.01,
    c_prior=0.01,
    c_dense=1.0,
    c_anatomical=0.6,
    c_grad=0.4,
    kernel_distance=27,  # 18,9
    kernel_radius=16,
    quiet=False,
    img_names=["frontal"],
    shapes_to_skip_fitting=["None"],
    plot_freq=250,
    plot_tag="",
    rotation=[0],
    **kwargs,
  ): 
    self.ssam = ssam_obj

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
    self.img_names = img_names
    self.shapes_to_skip_fitting = shapes_to_skip_fitting
    self.plot_freq = plot_freq  # frequency to show debug plots
    self.plot_tag = plot_tag  # file name for debug plots

    self.lobes = ["RUL", "RML", "RLL", "LUL", "LLL"]

    # info from training dataset
    self.lmOrder = lmOrder
    self.shape = copy(shape)
    self.shapenorms = normals
    self.xRay = xRay  # XR edge map
    self.projLM = None
    self.projLM_ID = None
    self.fissureLM_ID = None
    self.model = model
    self.transform = transform[np.newaxis]
    # appearance model inputs
    self.density = density  # landmark densities

    # number of images used for reconstruction (assumes img shape = 500 x 500)
    self.number_of_imgs = max(np.sum(img.shape) - 1000, 1)
    self.rotation = rotation

    # format x-ray for image enhancement
    self.img = copy(img)  # XR image array
    img_scaled = self.rescale_image_intensity(copy(img))
    img_scaled *= 0.999  # avoid floating point error in scalar causing img > 1
    self.img_scaled = copy(img_scaled)

    img_local = utils.localNormalisation(copy(img_scaled), 20)
    img_local = self.rescale_image_intensity(img_local)
    self.img_local = np.round(img_local, 4)
    
    self.imgCoords = imgCoords
    self.imgCoords_axes = imgCoords_axes

    self.optIter = 0
    self.optStage = "align"

    self.scale = 1

    self.bounds_index_scale = kwargs["bounds_index_scale"]
    self.bounds_index_shape = kwargs["bounds_index_shape"]

    self.num_modes = modeNum

    if type(shape) == dict:
      # initialise shape parameters for each sub-shape, to reduced mode nums
      self.model_s = dict.fromkeys(model.keys())
      self.model_g = dict.fromkeys(model.keys())
      self.b = copy(ssam_obj.model_parameters)[:self.num_modes]
      self.variance = ssam_obj.variance[:self.num_modes]
      self.std = ssam_obj.std[:self.num_modes]

      for k in model.keys():
        print(k)
        # get shape for reshaping
        if self.density.ndim == 3:
          number_of_features = len(shape[k][0]) + len(self.density[0][0])
        else:
          number_of_features = len(shape[k][0]) + 1  # one value for density
        # shape model components only
        self.model_s[k] = self.filterModelShapeOnly(
          self.model[k][:modeNum], number_of_features
        )
        # gray-value model components only
        self.model_g[k] = self.filterModelDensityOnly(
          self.model[k][:modeNum], number_of_features
        )

      self.meanScaled = self.stackShapeAndDensity(
        self.scaleShape(shape["ALL"]), self.density.mean(axis=0)
      )
    else:
      raise AttributeError("Unexpected data type")
      # # get shape for reshaping
      # if self.density.ndim == 3:
      #   number_of_features = len(shape[0]) + len(self.density[0])
      # else:
      #   number_of_features = len(shape[0]) + 1  # one value for density
      # # parameters
      # self.b = np.zeros(modeNum)
      # # shape model components only
      # self.model_s = self.filterModelShapeOnly(
      #   self.model[:modeNum], number_of_features
      # )
      # # gray-value model components only
      # self.model_g = self.filterModelDensityOnly(
      #   self.model[:modeNum], number_of_features
      # )
      # self.meanScaled = self.stackShapeAndDensity(
      #   self.scaleShape(shape), self.density.mean(axis=0)
      # )

  def rescale_image_intensity(self, img):
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

  def loss_function(self, pose, scale=None, b=None):
    # call initialised variables from __init__
    xRay = self.xRay
    shape = copy(self.shape)
    shapenorms = self.shapenorms
    projLM = self.projLM
    projLM_ID = self.projLM_ID
    fissureLM_ID = self.fissureLM_ID
    self.optIter += 1

    prior = []
    # copy mean shape before shape is adjusted
    meanShape = copy(self.shape["ALL"])
    # call test parameters from optimizer
    self.b = copy(b)

    if not self.quiet:
      print(f"\nNext loop iter {self.optIter}")
      print("\t\t opt params ", pose, scale)
      print("\t\t\t", b)
    # apply new shape parameters to each lobe
    all_morphed = self.morphAirway(
      meanShape,
      self.b,
      self.model_s["ALL"][: len(self.b)],
    )

    if pose.size == 2:
      pose = np.insert(pose, 1, 0)
    align = np.mean(all_morphed, axis=0)

    all_morphed = self.centerThenScale(all_morphed, scale, align)
    # apply transformation to shape
    all_morphed = all_morphed + pose

    # print("pose and scale off")
    airway_morphed = all_morphed[self.lmOrder["Airway"]]
    # check shape has not moved to be larger than XR or located outside XR
    outside_bounds = np.any(
      (all_morphed[:, 0].max() > self.imgCoords[:, 0].max())
      | (all_morphed[:, 0].min() < self.imgCoords[:, 0].min())
      | (all_morphed[:, 1].max() > self.imgCoords[:, 1].max())
      | (all_morphed[:, 1].min() < self.imgCoords[:, 1].min())
      | (all_morphed[:, 2].min() < self.imgCoords[:, 2].min())
    )

    self.scale = copy(scale)  # set globally to call in fitTerm
    lobe_morphed = dict.fromkeys(self.lobes)
    for lobe in self.lobes:
      lobe_morphed[lobe] = all_morphed[self.lmOrder[lobe]]

    # get losses
    fit = self.fitTerm(xRay, lobe_morphed)
    loss_anatomicalShadow = 0.0
    if self.number_of_imgs > 1:
      density_t = [None] * self.number_of_imgs
      for i, axes_i in enumerate(self.imgCoords_axes):
        if self.rotation[i] == 0:
          all_morphed_rot = copy(all_morphed)
          airway_morphed_rot = copy(airway_morphed)
        elif self.rotation[i] == 45:
          all_morphed_rot = utils.rotate_coords_about_z(all_morphed, 45)
          airway_morphed_rot = utils.rotate_coords_about_z(airway_morphed, 45)
        else:
          all_morphed_rot = utils.rotate_coords_about_z(all_morphed, self.rotation[i])
          airway_morphed_rot = utils.rotate_coords_about_z(airway_morphed, self.rotation[i])
        density_t[i] = self.get_density(
          all_morphed_rot[:, axes_i], self.img[i], self.imgCoords[:, axes_i]
        )
        # can turn off anatomical shadow by setting coeff to 0
        # coeff/2 as we want to keep weighting same relative to other losses
        if self.c_anatomical != 0.0:
          if i == 0:
            loss_anatomicalShadow += (
              self.c_anatomical
              / 1.0
              * self.compute_anatomical_shadow_loss(
                  self.img_local[i],
                  self.imgCoords[:, axes_i],
                  airway_morphed_rot,
                  self.lmOrder,
                  self.projLM_ID_multipleproj[i]["Airway"],
                  kernel_distance=self.kernel_distance,
                  kernel_radius=self.kernel_radius,
                  axes=axes_i,
                  debug_fname=f"iter{self.optIter}proj{i}shadow.png",
              )
            )
      # convert list to array
      # change from shape N_imgs, N_lms -> N_lms, N_imgs
      # density_t = np.array(density_t).T
      density_t = np.stack(density_t, axis=1)
    else:
      density_t = self.get_density(
        all_morphed, self.img, self.imgCoords, self.imgCoords_axes[0]
      ).reshape(-1, 1)

      if self.c_anatomical != 0.0:
        loss_anatomicalShadow += self.c_anatomical * self.compute_anatomical_shadow_loss(
          self.img_local,
          self.imgCoords[:, self.imgCoords_axes[0]],
          airway_morphed,
          self.lmOrder,
          self.projLM_ID["Airway"],
          kernel_distance=self.kernel_distance,
          kernel_radius=self.kernel_radius,
          axes=self.imgCoords_axes[0],
          debug_fname=f"iter{self.optIter}shadow.png",
        )

    scaled_morphed_shape_appearance = self.stackShapeAndDensity(self.scaleShape(all_morphed), density_t)
    prior = self.priorTerm(scaled_morphed_shape_appearance, self.meanScaled)

    densityFit = self.densityLoss(
      density_t.reshape(-1),
      self.density.mean(axis=0).reshape(-1),
      self.model_g["ALL"][: len(self.b)],
      self.b,
    )
    if not self.quiet:
      print("\tfit loss", fit)
      print("\tdensity loss", densityFit)
      print("\tprior loss", prior)  # round(prior,4))

    # add all loss contributions together
    loss = (
      (self.c_prior * prior) + (self.c_dense * densityFit) + (self.c_edge * fit)
    )
    loss += loss_anatomicalShadow

    if outside_bounds and self.optIter > 100:
      # as we use max value from loss log, make sure this is only after 100
      #   iterations, so the max value will not be sensitive to outliers
      loss += max(self.lossLog) * 0.2
      if not self.quiet:
        print("OUTSIDE OF BOUNDS")
    if (self.optIter % 1000) == 0:
      print(f"\titer is {self.optIter}. total loss {loss}")

    # return debugging plot of outline overlaid on image
    if (self.optIter % self.plot_freq == 0) and not self.quiet:
      if self.number_of_imgs == 1:
        self.overlayAirwayOnXR(self.img, all_morphed, scale)
      elif self.number_of_imgs >= 2:
        self.overlayAirwayOnXR_multipleimgs(self.img, all_morphed, scale)
    assert ~np.isnan(densityFit), "unexpected NaN {}".format(density_t)
    return loss

  def compute_anatomical_shadow_loss(
    self,
    img,
    img_coords,
    landmarks,
    lmOrder,
    projLM_ID,
    kernel_distance,
    kernel_radius,
    axes=[0, 2],
    debug_fname="shadow.png"
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
      -img.shape[1] / 2.0 * self.spacing_xr[0],
      img.shape[1] / 2.0 * self.spacing_xr[0],
      -img.shape[0] / 2.0 * self.spacing_xr[2],
      img.shape[0] / 2.0 * self.spacing_xr[2],
    ]

    skeleton_ids = lmOrder["SKELETON"]
    airway_ids = lmOrder["Airway"]
    # get airway points not on skeleton (surface only)
    airway_surf_ids = airway_ids[~np.isin(airway_ids, skeleton_ids)]
    # get surface points that are in projected points list
    airway_surf_ids = airway_surf_ids[np.isin(airway_surf_ids, projLM_ID)]

    skel_pts = landmarks[skeleton_ids][:, axes]
    silhouette_pts = landmarks[airway_surf_ids][:, axes]

    dists = cdist(silhouette_pts, skel_pts)
    nearest_skel_pt = np.argmin(dists, axis=1)
    vec = silhouette_pts - skel_pts[nearest_skel_pt]
    div = np.sqrt(np.einsum("ij,ij->i", vec, vec))
    norm_vec = np.divide(vec, np.c_[div, div])

    all_p_in = (
      silhouette_pts + norm_vec * kernel_distance * self.spacing_xr[axes]
    )
    all_p_out = (
      silhouette_pts - norm_vec * kernel_distance * self.spacing_xr[axes]
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
        draw.disk(center=(p_in_index[1], p_in_index[0]), radius=kernel_radius, shape=img.shape)
      ]
      c_out = img[
        draw.disk(center=(p_out_index[1], p_out_index[0]), radius=kernel_radius, shape=img.shape)
      ]

      energy_at_p = (c_in.mean() - c_out.mean()) / c_out.mean()
      if not np.isnan(energy_at_p):
        energy.append(energy_at_p)
      else:
        delInd.append(p)
    # print(energy)
    energy = np.array(energy)
    silhouette_pts = np.delete(silhouette_pts, delInd, axis=0)

    if self.optIter % 500 == 0 and not self.quiet:
      plt.close()
      # for debugging anatomical shadow values
      _, ax = plt.subplots()
      ax.imshow(img, cmap='gray', extent=extent)
      scatter = ax.scatter(silhouette_pts[:,0], silhouette_pts[:,1],
                            c=energy, s=2)
      plt.colorbar(scatter)
      plt.savefig(f'images/reconstruction/debug/{debug_fname}')

    # account for empty arrays when all points are outside of the domain
    if len(energy) == 0:
      return 0
    else:
      if not self.quiet:
        print("\tanatomicalShadow", energy.sum(), energy.mean())
      return (energy).mean()

  def scaleShape(self, shape):
    """
    return shape (lm x 3 array) with 0 mean and 1 std
    """
    return (shape - shape.mean(axis=0)) / shape.std()

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
    model_noApp = model_as_columns[:, :, :-number_of_appearances_cols]
    # reshape to 2D array
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
    model_no_shape = model_as_columns[:, :, -number_of_appearances_cols:]
    # reshape to 2D array
    return model_no_shape.reshape(model.shape[0], -1)

  def normaliseDist(self, dist):
    """
    Normalise a distance or list of distances to have range [0,1]
    """
    return np.exp(-1 * np.array(dist) / 5)

  def densityLoss(self, density_t, densityMean, model, b):
    """ """
    # modelled density
    density_m = self.ssam.morph_model(
      densityMean, model, b, self.num_modes
    )
    abs_diff = abs(density_t - density_m)
    return np.mean(abs_diff)

  def morphAirway(self, shape, shapeParams, model):
    """
    Adjust shape transformation and scale.
    Imports to SSM and extracts adjusted shape.
    """
    mean_to_align = shape.mean(axis=0)
    shape_centred = shape - mean_to_align
    scaler = shape_centred.std()
    shape_scaled = (shape_centred / scaler)

    shape_scaled_morphed = shape_scaled + np.dot(
      shapeParams[np.newaxis, :] * np.sqrt(self.variance[np.newaxis, :]), model
    ).reshape(-1, 3)
    # shape_scaled_morphed = self.ssam.morph_model(shape_scaled.reshape(-1), shapeParams[: self.num_modes], self.num_modes)

    shape_out = (shape_scaled_morphed * scaler) + mean_to_align

    morphed_displacement = utils.euclideanDist(shape_out, shape)
    if not self.quiet:
      print(
        "shape diff [real space] \t mean",
        np.mean(morphed_displacement),
        "\t max",
        np.max(morphed_displacement),
      )

    return shape_out

  def optimiseAirwayPoseAndShape(
    self, objective, init, bounds, epochs=2, threads=1
  ):
    """
    Minimises objective function using Nevergrad gradient-free optimiser
    """
    instrum = ng.p.Instrumentation(
      pose=ng.p.Array(init=init[:self.bounds_index_scale]).set_bounds(
        bounds[:self.bounds_index_scale, 0], bounds[:self.bounds_index_scale, 1]
      ),
      scale=ng.p.Scalar(init=init[self.bounds_index_scale]).set_bounds(
        bounds[self.bounds_index_scale, 0], bounds[self.bounds_index_scale, 1]
      ),
      b=ng.p.Array(init=np.zeros(self.b.size)).set_bounds(
        bounds[self.bounds_index_shape, 0], bounds[self.bounds_index_shape, 1]
      ),
    )

    optimizer = ng.optimizers.NGO(  # CMA(#NGO(
      parametrization=instrum, budget=epochs, num_workers=threads
    )

    # recommendation = optimizer.minimize(objective)
    self.lossLog = []
    recommendation = optimizer.provide_recommendation()
    for _ in range(optimizer.budget):
      x = optimizer.ask()
      loss = objective(*x.args, **x.kwargs)
      optimizer.tell(x, loss)
      self.lossLog.append(loss)
    recommendation = (
      optimizer.provide_recommendation()
    )  # update recommendation

    tag = ""
    utils.plotLoss(
      self.lossLog,
      tag=self.plot_tag,
      wdir="images/reconstruction/",
    )  # plot loss

    optOut = dict.fromkeys(["pose", "scale", "b"])
    # optOut = dict.fromkeys(["pose","b"])
    optOut["pose"] = recommendation.value[1]["pose"]
    optOut["scale"] = recommendation.value[1]["scale"]
    optOut["b"] = recommendation.value[1]["b"]
    print("recommendation is", recommendation.value)

    return optOut

  def fitTerm(self, xRay, shapeDict):

    scaler = 5.0  # distance weighting factor
    num_points = 0  # initialise number of points
    fit_list = [] 
    for img_index, axes in zip(range(0, self.number_of_imgs), self.imgCoords_axes):
      if self.rotation[img_index] == 45: 
        continue
      if self.number_of_imgs == 1:
        projLM_ID_i = self.projLM_ID.copy()
        if type(xRay) == list:
          # below line of code only needed if two outlines given by accident in json
          xRay_i = xRay[0].copy()
        elif type(xRay) == np.ndarray:
          xRay_i = xRay.copy()
        else:
          raise AttributeError(f"unrecognised variable type for xRay in fitTerm, {type(xRay)}. ")
      else:
        projLM_ID_i = self.projLM_ID_multipleproj[img_index]
        xRay_i = xRay[img_index]
      for k, key in enumerate(self.lobes):
        # skip certain lobes or sub-shapes for different images
        # as they may be non-visible
        if key not in self.shapes_to_skip_fitting[img_index]:
          shape = utils.rotate_coords_about_z(shapeDict[key], self.rotation[img_index])[projLM_ID_i[key]][:, axes]
          num_points += len(shape)

          fit_list.append(self.fitLoss(shape, xRay_i, scaler))

    loss_fit = (1.0 / (num_points * self.number_of_imgs)) * np.sum(fit_list)

    return loss_fit

  def normalisedDistance(self, shape1, shape2, scaler=5):
    """
    Find closest normalised distance between two point clouds
    """
    distArr = cdist(shape1, shape2)
    closest_dist = np.min(distArr, axis=1)
    return np.exp(-closest_dist / scaler)

  def fitLoss(self, shape, xRay, scaler):
    # get distance term (D_i)
    dist = self.normalisedDistance(shape, xRay, scaler)
    fit_term_loss_i = abs(1 - dist)
    return np.sum(fit_term_loss_i)

  def priorTerm(self, shape, meanShape):
    """
    Compare shape generated by optimisation of shape parameters
    with mean shape of dataset, using mahalanobis distance.
    """
    loss_prior = (
      np.sum(utils.mahalanobisDist(meanShape, shape)) / meanShape.shape[0]
    )

    return loss_prior

  def overlayAirwayOnXR(self, img, coords, scale, tag=""):
    extent = [
      self.imgCoords[:, self.imgCoords_axes[0][0]].min(),
      self.imgCoords[:, self.imgCoords_axes[0][0]].max(),
      self.imgCoords[:, self.imgCoords_axes[0][1]].min(),
      self.imgCoords[:, self.imgCoords_axes[0][1]].max(),
    ]
    plt.close()
    plt.imshow(img, cmap="gray", extent=extent)
    if type(self.xRay) == list:
      # below line of code only needed if two outlines given by accident in json
      xRay_i = self.xRay[0].copy()
    elif type(self.xRay) == np.ndarray:
      xRay_i = self.xRay.copy()
    plt.scatter(xRay_i[:, 0], xRay_i[:, 1], s=4, c="black")
    for key in self.projLM_ID.keys():
      projLM_key = self.projLM_ID[key]
      plt.scatter(
        coords[self.lmOrder[key]][projLM_key, 0],
        coords[self.lmOrder[key]][projLM_key, 2],
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
      "images/reconstruction/debug/iter{}{}.png".format(str(self.optIter), tag),
      bbox_inches="tight",
      pad_inches=0,
    )
    return None

  def overlayAirwayOnXR_multipleimgs(self, img, coords, scale, tag=""):
    for i, (im_i, axes, angle) in enumerate(zip(img, self.imgCoords_axes, self.rotation)):
      extent = [
        self.imgCoords[:, axes[0]].min(),
        self.imgCoords[:, axes[0]].max(),
        self.imgCoords[:, axes[1]].min(),
        self.imgCoords[:, axes[1]].max(),
      ]
      plt.close()
      plt.imshow(im_i, cmap="gray", extent=extent)
      if self.rotation[i] != 45: 
        plt.scatter(self.xRay[i][:, 0], self.xRay[i][:, 1], s=4, c="black")
      coords_to_plot = utils.rotate_coords_about_z(coords, angle)
      for key in self.projLM_ID.keys():
        projLM_index = self.projLM_ID_multipleproj[i][key]
        if key not in self.shapes_to_skip_fitting[i]:
          plt.scatter(
            coords_to_plot[self.lmOrder[key]][projLM_index, axes[0]],
            coords_to_plot[self.lmOrder[key]][projLM_index, axes[1]],
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
        "images/reconstruction/debug/img{}-iter{}{}.png".format(
          i, str(self.optIter), tag
        ),
        bbox_inches="tight",
        pad_inches=0,
      )
    return None

  def getProjectionLandmarks(self, faceIDs, faceNorms, points, plane=1):
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
              np.min(faceNorms[shape][norms[pID]][:, plane]) < 0
              and np.max(faceNorms[shape][norms[pID]][:, plane]) > 0
            ):
              projectionLM[shape].append(points[shape][pID])
              projectionLM_ID[shape].append(pID)
            # else:
            #   print(np.array(faceNorms[shape][norms[pID]][:,1]),
            #         len(faceNorms[shape][norms[pID]][:,1]))

        projectionLM[shape] = np.array(projectionLM[shape])
        # delete projection plane from coords
        projectionLM[shape] = np.delete(projectionLM[shape], plane, axis=1)
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
            np.min(faceNorms[norms[pID]][:, plane]) < 0
            and np.max(faceNorms[norms[pID]][:, plane]) > 0
          ):
            projectionLM.append(points[pID])
            projectionLM_ID.append(pID)
        else:
          continue
      np.where(
        np.isin(faceIDs[:, 0], points)
        | (faceIDs[:, 1] == pID)
        | (faceIDs[:, 2] == pID)
      )[0]

      projectionLM = np.array(projectionLM)
      # delete projection plane from coords
      projectionLM = np.delete(projectionLM, plane, axis=1)
      projectionLM_ID[shape] = np.array(projectionLM_ID[shape])
    return projectionLM, projectionLM_ID

  def deleteShadowedEdges(self, coords, projLM, projLM_ID):
    """
    Deletes landmarks that are not found on the radiograph
    (overlapped by spine or fissures that are not visible)
    """
    shape = copy(coords)

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
        # filter low points
        # (corresponds to curvature at RUL RLL intersection)
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
        del delInd  # delete to avoid accidental deletion of wrong indexes

        """filter outlier clusters by applying 75th percentile 
            mahalanobis distance filter twice"""
        tmpprojLM = copy(projLM[key])  # store initial values
        for i in range(2):
          # keep data to outer side of RUL
          dataKept = projLM["RUL"][
            np.where(
              projLM["RUL"][:, 0]
              < (
                projLM["RUL"][:, 0].min()
                + (projLM["RUL"][:, 0].max() - projLM["RUL"][:, 0].min()) / 2
              )
            )
          ]
          # set aside inner coordinates for filtering
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
          # set coords with high mean mahalanobis distance for deletion
          delInd = np.unique(
            np.where(md.mean(axis=0) > np.percentile(md.mean(axis=0), 65))[0]
          )

          data = np.delete(data, delInd, axis=0)
          del delInd  # delete to avoid accidental deletion of wrong indexes
          projLM[key] = np.vstack((dataKept, data))  # reset array
        # loop to find landmark ID's removed
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

        # delete previous stored indexes
        projLM[key] = np.delete(projLM[key], delInd, axis=0)
        projLM_ID[key] = np.delete(projLM_ID[key], delInd)

        # filter out upper right points in the RLL
        # set coords with high mean mahalanobis distance for deletion
        delInd = np.unique(
          np.where(md.mean(axis=0) > np.percentile(md.mean(axis=0), 92.5))[0]
        )
        projLM[key] = np.delete(projLM[key], delInd, axis=0)
        projLM_ID[key] = np.delete(projLM_ID[key], delInd)

        for i in range(2):
          # find lowest point in RLL to filter out
          # this is needed as it is typically not segmented by XR seg tool
          cornerP = projLM["RLL"][np.argmin(projLM["RLL"][:, 1], axis=0)]
          cutoff = 0.4  # define cutoff distance

          # use mahalanobis distance to filter out points
          projLM_ID["RLL"] = projLM_ID["RLL"][
            np.where(utils.mahalanobisDist(projLM["RLL"], cornerP) > cutoff)
          ]
          projLM["RLL"] = projLM["RLL"][
            np.where(utils.mahalanobisDist(projLM["RLL"], cornerP) > cutoff)
          ]
      elif key == "LUL":  # if left lung
        delInd = np.where(
          (
            (projLM[key][:, 0] - projLM[key][:, 0].min() < 0.4 * width)
            & (projLM[key][:, 1] - projLM[key][:, 1].min() < 0.8 * height)
          )
          | (projLM[key][:, 1] - projLM[key][:, 1].min() < 0.4 * height)
        )
        projLM[key] = np.delete(projLM[key], delInd, axis=0)
        projLM_ID[key] = np.delete(projLM_ID[key], delInd)
      else:
        projLM[key] = np.delete(projLM[key], delInd, axis=0)
        projLM_ID[key] = np.delete(projLM_ID[key], delInd)
    return projLM, projLM_ID

if __name__ == "__main__":
  print(__doc__)
