"""
"""

import argparse
from copy import copy
from glob import glob
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pyssam
import vedo as v

import userUtils as utils


class RespiratorySSAM(pyssam.StatisticalModelBase):
  def __init__(
    self,
    lm,
    lm_ct,
    imgs,
    imgsOrigin,
    imgsSpacing,
    rotation=[0],
    **kwargs,
  ):
    # check all input datasets have same number of samples
    assert (
      lm.shape[0]
      == imgs.shape[0]
      == imgsOrigin.shape[0]
      == imgsSpacing.shape[0]
    ), "non-matching dataset size"

    self._num_landmarks = lm.shape[1]

    # -initialise input variables
    self.lm = lm  # centered landmarks from GAMEs
    self.lm -= self.lm.mean()
    self.lm_ct = lm_ct  # landmarks transformed to same coordinate frame as CT

    # shape modelling classes
    self.ssm = pyssam.SSM(lm)
    # appearance model classes. 
    # if two images are provided, stack the landmarked density at each
    if imgs.ndim == 3:
      img_axes_i = [0, 2]
      appearance_xr_utils = pyssam.utils.AppearanceFromXray(
        imgs, 
        imgsOrigin[:, img_axes_i], 
        imgsSpacing[:, img_axes_i]
      )
      img_coords = appearance_xr_utils.pixel_coordinates
      appearance = appearance_xr_utils.all_landmark_density(lm_ct[:, :, img_axes_i])
      self.sam = pyssam.SAM(appearance)
      self.density = self.sam.appearance_scale
    elif imgs.ndim == 4:
      img_axes = kwargs["img_coords_axes"]
      density_all = []
      reshape_to_shape = lm_ct.shape[:-1]+tuple([1])
      img_coords = []
      for proj_i, img_axes_i in enumerate(kwargs["img_coords_axes"]):
        rot_coords = []
        for i, patient in enumerate(lm_ct):
          # centre to rotate points around
          img_centre = imgsOrigin[i] + (imgsSpacing[i]*500.)/2
          rot_coords_i = utils.rotate_coords_about_z(patient, rotation[proj_i], img_centre)
          rot_coords.append(rot_coords_i)
        rot_coords = np.array(rot_coords)
        appearance_xr_utils = pyssam.utils.AppearanceFromXray( 
          imgs[:, proj_i], 
          imgsOrigin[:, img_axes_i], 
          imgsSpacing[:, img_axes_i]
        )
        img_coords.append(appearance_xr_utils.pixel_coordinates)
        appearance = appearance_xr_utils.all_landmark_density(rot_coords[:, :, img_axes_i])
        self.sam = pyssam.SAM(appearance)
        appearance_scale = self.sam.appearance_scale.copy().reshape(reshape_to_shape)
        density_all.append(appearance_scale)
      self.density = np.dstack(density_all)

    # -initialise appearance model data
    self.imgs = imgs
    self.imgCoords = np.array(img_coords)

    self.shape_appearance = np.dstack((self.ssm.landmarks_columns_scale.reshape(self.lm.shape), self.density))
    self.shape_appearance_columns = self.landmark_data_to_column(self.shape_appearance)

  def compute_dataset_mean(self) -> np.array:
    return np.mean(self.shape_appearance_columns, axis=0)
