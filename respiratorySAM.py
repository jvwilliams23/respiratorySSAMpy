"""
 User args:
  --inp
    set landmark directory
  --drrs
    set radiograph directory (drr = digitally reconstructed radiograph)
  --debug
    use debug flags
  --getModes
    write main modes of variation to surface

  Develops SAM based on landmarks determined by GAMEs algorithm
  and DRRs extracted from the same CT dataset.

  @author: Josh Williams
"""
# Note to future developers:
#   SAM has been developed to be generic, i.e. that it can easily be
#   implemented for appearance modelling of other organs, or images in general.
#   When adapting to other organs, images etc, some changes may be required
#   such as user arguments and data structure (file names etc).


import argparse
import random
from copy import copy
from glob import glob
from sys import argv, exit
from time import time

import matplotlib.pyplot as plt
import numpy as np
import vtk
from scipy import ndimage
from scipy.spatial.distance import cdist, pdist
from skimage import io
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from vedo import *

# plot graphs for compactness, specificity etc
# from ssmPlot import *
# from userUtils import cm2inch, doPCA, trainTestSplit
import userUtils as utils


class RespiratorySAM:
  def __init__(
    self, lm, imgs, imgsOrigin, imgsSpacing, train_size=0.9, axes=[0, 2]
  ):

    # check all input datasets have same number of samples
    assert (
      lm.shape[0]
      == imgs.shape[0]
      == imgsOrigin.shape[0]
      == imgsSpacing.shape[0]
    ), "non-matching dataset size"

    # -import functions
    self.doPCA = utils.doPCA
    self.trainTestSplit = utils.trainTestSplit

    # -initialise variables
    self.lm = lm
    self.imgs = imgs
    self.imgsN = self.normaliseImageDensity(self.imgs)
    self.imgCoords_all = self.drrArrToRealWorld(
      self.imgs, imgsOrigin, imgsSpacing
    )
    # filter selected axes (x,z for frontal or y,z for lateral)
    self.imgCoords = self.imgCoords_all[:, :, axes]

    # test landmarks are within image
    # self.testLandmarkAndImageBounds(self.lm[:,:,[0,2]], self.imgCoords)

    self.density_base = self.getTrainingDensity(
      self.lm, self.imgs, self.imgCoords
    )

    self.density = (
      self.density_base - self.density_base.mean(axis=1)[:, np.newaxis]
    )
    self.density = self.density / self.density.std(axis=1)[:, np.newaxis]
    self.g_train = self.density

    if __name__ == "__main__":
      self.pca, k_95 = self.doPCA(
        self.density, 0.95
      )  # -train the model and get PCs
      self.phi = self.pca.components_
      self.variance = self.pca.explained_variance_
      self.std = np.sqrt(self.variance)

    # -initialise empty vector. Can only be same size as training set
    self.b = np.zeros(self.g_train.shape[0])

  def testLandmarkAndImageBounds(self, lm, imgCoords):
    minCheck = imgCoords.min(axis=1)[:, 0] < lm.min(axis=1)[:, 0]
    maxCheck = imgCoords.max(axis=1)[:, 0] > lm.max(axis=1)[:, 0]
    assert maxCheck.all() and minCheck.all(), "shape is outside image bounds"

  def landmarksToRealWorld(self, lm, trans):
    return lm + trans

  def drrArrToRealWorld(self, img, origin, spacing):
    """
    Set bottom left coordinate to CT origin, and assign real world coord to DRR
    """
    # images all square so should have same shape and can be initialised same way
    xCoords = np.zeros((origin.shape[0], img.shape[-2]))
    yCoords = np.zeros((origin.shape[0], img.shape[-2]))
    zCoords = np.zeros((origin.shape[0], img.shape[-1]))

    xBase = np.linspace(0, img.shape[-2], img.shape[-2])
    yBase = np.linspace(0, img.shape[-2], img.shape[-2])
    zBase = np.linspace(0, img.shape[-1], img.shape[-1])
    if origin.ndim == 2:
      xCoords = (
        origin[:, 0]
        + np.meshgrid(xBase, np.ones(spacing[:, 0].size))[0].T * spacing[:, 0]
      )
      yCoords = (
        origin[:, 1]
        + np.meshgrid(yBase, np.ones(spacing[:, 1].size))[0].T * spacing[:, 1]
      )
      zCoords = (
        origin[:, 2]
        + np.meshgrid(zBase, np.ones(spacing[:, 2].size))[0].T * spacing[:, 2]
      )
    elif origin.ndim == 1:
      xCoords = (
        origin[0]
        + np.meshgrid(xBase, np.ones(spacing[0].size))[0].T * spacing[0]
      )
      yCoords = (
        origin[1]
        + np.meshgrid(xBase, np.ones(spacing[1].size))[0].T * spacing[1]
      )
      zCoords = (
        origin[2]
        + np.meshgrid(zBase, np.ones(spacing[2].size))[0].T * spacing[2]
      )
    else:
      printc("unexpected origin dimensions in SAM.drrArrToRealWorld")

    return np.dstack(
      (
        np.swapaxes(xCoords, 0, 1),
        np.swapaxes(yCoords, 0, 1),
        np.swapaxes(zCoords, 0, 1),
      )
    )

  # def pixelisePointCloud(lm, density, imgCoords):
  #   img = np.zeros((500,500))
  #   nearestX = np.argmin(abs(imgCoords[:,0].reshape(-1,1)
  #                            -lm[:,0]),
  #                            axis=0)
  #   nearestZ = np.argmin(abs(imgCoords[:,1].reshape(-1,1)
  #                             -lm[:,2]),
  #                            axis=0)
  #   img[499-nearestZ, nearestX] = density

  def getDensity(self, lm, img, imgCoords):
    """
    Returns density of a landmark based on comparing landmark coordinates to
    pixel with nearest real world coordinate in x and z direction.

    Inputs:
        lm: (coords x 3) array of landmarks
        img: (drr x-dimension x drr y-dimension) array of grey values
        imgCoords: (drr x-dimension x 2)
            Note that imgCoords axis=1 value assumes a square figure
            (same number of x and y pixels)

    Return:
        density: (coords) array of grey value for each landmarks
    """
    # -use argmin to find nearest pixel neighboring a point
    nearestX = np.argmin(abs(lm[:, 0] - imgCoords[:, 0].reshape(-1, 1)), axis=0)
    nearestZ = np.argmin(abs(lm[:, 2] - imgCoords[:, 1].reshape(-1, 1)), axis=0)

    return img[len(img) - 1 - nearestZ, nearestX]  # gives correct result

  def getTrainingDensity(self, lm, img, imgCoords):
    """
    Returns density of a landmark based on comparing landmark coordinates to
    pixel with nearest real world coordinate in x and z direction.

    Inputs:
        lm: (patients x coords x 3) array of landmarks
        img: (patients x drr x-dimension x drr y-dimension) array of grey values
        imgCoords: (patients x drr x-dimension x 2)
            Note that imgCoords axis=1 value assumes a square figure
            (same number of x and y pixels)

    Return:
        density: (patients x coords) array of grey value for each landmark
    """

    dshape = list(lm.shape[:-1])
    density = np.zeros(dshape)
    debug = False

    for p in range(lm.shape[0]):
      if debug:
        print("testing {} patient {}".format(shape, patientIDs[p]))
      density[p] = self.getDensity(lm[p], img[p], imgCoords[p])

      # if debug:
      #   plt.close()
      #   fig, ax = plt.subplots(1,2)
      #   ax[0].imshow(img[p], cmap='gray')
      #   ax[1].imshow(img[p], cmap='gray')
      #   ax[1].scatter(nearestX, 500-nearestZ,
      #                 c=density[p], cmap='gray', s=1, alpha=0.25)
      #   fig.savefig("imagesSAM/"+shape+"/SAM-norm-"+patientIDs[p]+".png",
      #               pad_inches=0, format="png", dpi=300)

      #   plt.close()
      #   fig, ax = plt.subplots(1,2)
      #   ax[0].imshow(img[p], cmap='gray')
      #   ax[1].imshow(np.ones(img[p].shape), cmap='gray')
      #   ax[1].scatter(nearestX, 500-nearestZ,
      #                 c=density[p], cmap='gray', s=2)
      #   fig.savefig("imagesSAM/"+shape+"/SAM-norm-bw-"+patientIDs[p]+".png",
      #               pad_inches=0, format="png", dpi=300)

    return density

  def normaliseImageDensity(self, density_im):
    """
    Sets density to 0 mean and std deviation of 1 for all datasets
    """
    densityN_im = (
      density_im - density_im.mean(axis=(1, 2))[:, np.newaxis, np.newaxis]
    )
    densityN_im = (
      densityN_im / densityN_im.std(axis=(1, 2))[:, np.newaxis, np.newaxis]
    )

    # densityN_im = (density_im
    #                 -density_im.mean(axis=0)[np.newaxis,:,:]
    #                 )
    # densityN_im = densityN_im/density_im.std(axis=0)[np.newaxis,:,:]
    return densityN_im

  def normaliseTestImageDensity(self, density_im):
    """
    Sets density to 0 mean and std deviation of 1 for single test image
    """

    #######-------UNSURE ABOUT WHICH AXIS TO USE
    densityN_im = density_im - density_im.mean()
    densityN_im /= densityN_im.std()
    # densityN_im = (density_im
    #                 -density_im.mean(axis=0)[np.newaxis,:]
    #                 )
    # densityN_im = densityN_im/density_im.std(axis=0)[np.newaxis,:]
    return densityN_im

  def denormaliseLandmarkDensity(self, densityN_lm, density_im):
    """
    Sets normalised density back to initial value
    """
    if densityN_lm.ndim == 1:
      density_lm = densityN_lm * density_im.std()
      density_lm = density_lm + density_im.mean()
    elif densityN_lm.ndim == 2:
      density_lm = densityN_lm * density_im.std(axis=(1, 2))[:, np.newaxis]
      density_lm = density_lm + density_im.mean(axis=(1, 2))[:, np.newaxis]
    # if densityN_lm.ndim == 1:
    #   density_lm = densityN_lm * density_im.std()
    #   density_lm = density_lm + density_im.mean()
    # elif densityN_lm.ndim == 2:
    #   density_lm = densityN_lm*density_im.std(axis=0)[np.newaxis,:]
    #   density_lm = density_lm+density_im.mean(axis=0)[np.newaxis,:]

    else:
      printc(
        "error in denormaliseLandmarkDensity function, SAM.py",
        "unexpected dimensions of input variable",
        c="r",
      )
    return density_lm

  def testReconstruction(self, trainData, mean_tr, phi_tr, k, pca):
    if k < 1:
      print("LOW k found")
      k = np.where(np.cumsum(pca.explained_variance_ratio_) > k)[0][0]
      print("num components is ", k)

    r_k = 0  # initialise reconstruction error

    for n in range(trainData.shape[-1]):
      s_i = trainData[:, n]
      # b = (s_i - mean_tr) * phi_tr.T
      b = np.dot((s_i - mean_tr), phi_tr)
      # s_i_k = mean_tr + (phi_tr.T[:k] * b[:k])
      s_i_k = mean_tr + np.dot(phi_tr[:, :k], b[:k])

      r_k_i = np.sum(abs(s_i_k - s_i)) / len(
        s_i
      )  # reconstruction error for instance i

      if debug:
        # print("s_i_k")
        # print(s_i_k)
        print("r_k")
        print(r_k.shape)
        print("SHAPE CHECK")
        print(s_i.shape, s_i_k.shape)
        print("mean is")
        print(mean_tr.shape)
        print("phi shape")
        print(phi_tr.shape)
        print(phi_tr[:, :k].shape)
        print("b shape")
        print(b.shape)
        print(b[:k].shape)
        p = Points(s_i.reshape(-1, 3), c="b", r=6)
        p2 = Points(s_i_k.reshape(-1, 3), c="g", r=8)
        show(p, p2)
        exit()
      r_k += r_k_i

    # print("r_k is ", r_k, "\n n is",n)
    r_k /= n + 1
    # print("Reconstruction error is", r_k, r_k/max(mean_tr)*100.,"%" )

    return r_k / max(mean_tr) * 100.0

  def testGeneralisation(self, testData, mean_tr, phi_tr, k, pca):
    if k < 1:
      print("LOW k found")
      k = np.where(np.cumsum(pca.explained_variance_ratio_) > k)[0][0]
      print("num components is ", k)

    g_k = 0  # initialise reconstruction error

    for n in range(testData.shape[-1]):
      s_i = testData[:, n]
      # b = (s_i - mean_tr) * phi_tr.T
      b = np.dot((s_i - mean_tr), phi_tr)
      # s_i_k = mean_tr + (phi_tr.T[:k] * b[:k])
      s_i_k = mean_tr + np.dot(phi_tr[:, :k], b[:k])

      g_k_i = np.sum(abs(s_i_k - s_i)) / len(
        s_i
      )  # reconstruction error for instance i

      if debug:
        # print("s_i_k")
        # print(s_i_k)
        print("r_k")
        print(r_k.shape)
        print("SHAPE CHECK")
        print(s_i.shape, s_i_k.shape)
        print("mean is")
        print(mean_tr.shape)
        print("phi shape")
        print(phi_tr.shape)
        print(phi_tr[:, :k].shape)
        print("b shape")
        print(b.shape)
        print(b[:k].shape)
        p = Points(s_i.reshape(-1, 3), c="b", r=6)
        p2 = Points(s_i_k.reshape(-1, 3), c="g", r=8)
        show(p, p2)
        exit()
      g_k += g_k_i

    # print("r_k is ", r_k, "\n n is",n)
    g_k /= n + 1
    # print("Reconstruction error is", r_k, r_k/max(mean_tr)*100.,"%" )

    return g_k / max(mean_tr) * 100.0

  def testSpecificity(self, trainData, mean_tr, phi_tr, k, pca, N=20):
    """
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
    """
    if k < 1:
      print("LOW k found")
      k = np.where(np.cumsum(pca.explained_variance_ratio_) > k)[0][0]
      print("num components is ", k)

    closestList = []
    spec_k = np.zeros(N)  # initialise reconstruction error

    for count, n in enumerate(range(N)):

      # -set b to a vector with zero mean and 3 S.D. max
      b = np.random.default_rng().standard_normal(size=(len(phi_tr)))
      # -generate random shape
      s_i_k = (mean_tr + np.dot(phi_tr[:, :k], b[:k])).reshape(-1, 1)

      # -find closest shape in training set
      closestS = np.argmin(np.sum(abs(trainData - s_i_k), axis=0))
      s_i_k = s_i_k.reshape(-1)
      closestList.append(closestS)
      s_i_dash = trainData[:, closestS]

      # -difference between generated shape and closest match
      spec_k_i = np.sum(abs(s_i_k - s_i_dash)) / len(s_i_dash)

      if debug:
        print("spec_k")
        print(spec_k)
        print("SHAPE CHECK")
        print(s_i_dash.shape, s_i_k.shape)
        print("mean is")
        print(mean_tr.shape)
        print("phi shape")
        print(phi_tr.shape)
        print(phi_tr[:, :k].shape)
        print("b shape")
        print(b.shape)
        print(b[:k].shape)
        p = Points(s_i_dash.reshape(-1, 3), c="b", r=6)
        p2 = Points(s_i_k.reshape(-1, 3), c="g", r=8)
        show(p, p2)
        exit()
      spec_k[count] = spec_k_i
    spec_k /= max(mean_tr)
    spec_k *= 100.0

    # -return mean and standard deviation
    return np.mean(spec_k), np.std(spec_k)

  def getg_allModes(self, xBar, phi, b):
    """
    Return point cloud that has been adjusted by a specified shape vector (b)

      Args:
        xBar (n,) array: mean shape
        phi (sampleNum, n) array: PCA components
        b (sampleNum, ) array: shape vector to vary points by
    """
    return xBar + np.dot(phi.T, b)


def getInputs():
  parser = argparse.ArgumentParser(
    description="statistical appearance model class"
  )
  parser.add_argument(
    "-i", "--inp", default=False, type=str, help="input files (landmarks)"
  )
  parser.add_argument(
    "--drrs", default=False, type=str, help="input files (drr)"
  )
  parser.add_argument(
    "--getModes",
    default=False,
    type=str,
    help="write main modes of variation to figure",
  )
  parser.add_argument(
    "--debug",
    default=False,
    type=bool,
    help="debug mode -- shows print checks and blocks" + "plotting outputs",
  )

  args = parser.parse_args()
  landmarkDir = args.inp
  drrDir = args.drrs
  debugMode = args.debug
  getModes = args.getModes

  return landmarkDir, drrDir, debugMode, getModes


if __name__ == "__main__":
  print(__doc__)

  landmarkDir, drrDir, debug, getModes = getInputs()

  trainSplit = 0.9
  if not landmarkDir and not drrDir:
    print("ERROR: you must declare input data directories.")
    exit()

  printc("Appearance modelling", c="green")
  # -Get directories for DRR and landmark data
  originDirs = glob(drrDir + "/origins/origins/drr*.md")  # .sort()
  spacingDirs = glob(drrDir + "/*/drr*.md")  # .sort()
  imDirs = glob(drrDir + "/*/drr*.png")  # .sort()
  originDirs.sort()
  spacingDirs.sort()
  imDirs.sort()
  # -check that user has declared correct directory
  patientIDs = [i.split("/")[-1].replace(".png", "")[-4:] for i in imDirs]
  landmarkDirs = glob(landmarkDir + "/landmarks*.csv")
  # landmarkDirs = sorted(landmarkDirs,
  #                       key=lambda x: int(x.replace(".csv","")[-4:]))
  landmarkDirs.sort()
  lmIDs = [i.split("/")[-1].split("landmarks")[1][:4] for i in landmarkDirs]
  landmarkDirsOrig = glob("landmarks/manual-jw/landmarks*.csv")
  landmarkDirsOrig.sort()
  # -used to align drrs and landmarks
  # transDirs = glob( drrDir+"/transforms/transforms/allLandmarks/"
  #                             +"transformParams_case*"
  #                             +"_m_*.dat")
  # transDirs.sort()

  if (
    len(imDirs) == 0
    or len(originDirs) == 0
    or len(landmarkDirs) == 0
    or len(spacingDirs) == 0
    or len(landmarkDirsOrig) == 0
  ):
    print(
      "ERROR: The directories you have declared are empty.",
      "\nPlease check your input arguments.",
    )
    exit()

  delInd = []
  for i, imD in enumerate(imDirs):
    currID = imD.split(".")[-2][-4:]
    if currID not in lmIDs:
      delInd.append(i)

  for dId in delInd[::-1]:
    originDirs.pop(dId)
    spacingDirs.pop(dId)
    imDirs.pop(dId)

  # landmarkTrans = np.vstack([np.loadtxt(t, skiprows=1, max_rows=1)
  #                           for t in transDirs])
  # -read data
  origin = np.vstack([np.loadtxt(o, skiprows=1)] for o in originDirs)
  spacing = np.vstack([np.loadtxt(o, skiprows=1)] for o in spacingDirs)
  # -load x-rays into a stacked array,
  # -switch so shape is (num patients, x pixel, y pixel)
  drrArr = np.rollaxis(np.dstack([utils.loadXR(o) for o in imDirs]), 2, 0)
  nodalCoords = np.array(
    [np.loadtxt(l, delimiter=",", skiprows=1) for l in landmarkDirs]
  )
  nodalCoordsOrig = np.array(
    [
      np.loadtxt(l, delimiter=",", skiprows=1, usecols=[1, 2, 3])
      for l in landmarkDirsOrig
    ]
  )
  carinaArr = nodalCoordsOrig[:, 1]

  # -offset centered coordinates to same reference frame as CT data
  lmProj = (
    nodalCoords + carinaArr[:, np.newaxis]
  )  # landmarkTrans[:, np.newaxis]
  # -create appearance model instance and load data
  sam = RespiratorySAM(lmProj, drrArr, origin, spacing, train_size=trainSplit)
  sam.testLandmarkAndImageBounds(sam.lm[:, :, [0, 2]], sam.imgCoords)
  drrPos = sam.imgCoords
  # drrArrNorm = sam.imgsN
  density = sam.g_train
  phi = sam.phi

  if getModes:
    printc("Plotting modes of appearance variation...", c="blue")
    modes = [0, 1, 2]
    stds = [3, -3]  # [10, 3, 0, -3, -10]
    meanShape = lmProj.mean(axis=0)
    meanImage = drrArr.mean(axis=0)
    plt.close()
    fig, ax = plt.subplots(nrows=len(stds), ncols=len(modes))  # ,
    # figsize=utils.cm2inch(17,10))
    for i, mode in enumerate(modes):
      ax[0][i].set_title("mode " + str(mode + 1), fontsize=11)
      for j, std in enumerate(stds):
        if i == 0:
          if std != 0:
            title = "SD = " + str(std)
          else:
            title = "mean"
          ax[j][0].set_ylabel(title, rotation=90, fontsize=11)
        b = sam.b
        b[mode] = std
        testIm = density.mean(axis=0) + (phi.T * b * sam.std)[:, mode]

        a = ax[j][i].scatter(
          meanShape[:, 0],
          meanShape[:, 2],
          c=(phi.T * b * sam.std)[:, mode],
          cmap="seismic",  #'RdBu_r',
          vmin=-3,
          vmax=3,
          s=1,
        )

        ax[j][i].xaxis.set_ticklabels([])
        ax[j][i].yaxis.set_ticklabels([])
        ax[j][i].spines["top"].set_visible(False)
        ax[j][i].spines["right"].set_visible(False)
        ax[j][i].spines["bottom"].set_visible(False)
        ax[j][i].spines["left"].set_visible(False)
        ax[j][i].get_xaxis().set_ticks([])
        ax[j][i].get_yaxis().set_ticks([])
    fig.colorbar(a, ax=ax.ravel().tolist())
    fig.text(
      0.5,
      0.95,
      "appearance components",
      horizontalalignment="center",
      fontsize=12,
    )
    fig.savefig(
      "images/SAM/SAM-densityModes-diff-meanShape.png",
      pad_inches=0,
      format="png",
      dpi=300,
    )

    plt.close()
    fig, ax = plt.subplots(nrows=len(stds), ncols=len(modes))  # ,
    for i, mode in enumerate(modes):
      ax[0][i].set_title("mode " + str(mode + 1), fontsize=11)
      for j, std in enumerate(stds):
        if i == 0:
          if std != 0:
            title = "SD = " + str(std)
          else:
            title = "mean"
          ax[j][0].set_ylabel(title, rotation=90, fontsize=11)
        b = sam.b
        b[mode] = std
        modeIm_N = density.mean(axis=0) + (phi.T * b * sam.std)[:, mode]
        modeIm = sam.denormaliseLandmarkDensity(modeIm_N, drrArr)
        a = ax[j][i].scatter(
          meanShape[:, 0],
          meanShape[:, 2],
          c=modeIm,
          cmap="gray",
          vmin=0,
          vmax=1,
          s=1,
        )

        ax[j][i].xaxis.set_ticklabels([])
        ax[j][i].yaxis.set_ticklabels([])
        ax[j][i].spines["top"].set_visible(False)
        ax[j][i].spines["right"].set_visible(False)
        ax[j][i].spines["bottom"].set_visible(False)
        ax[j][i].spines["left"].set_visible(False)
        ax[j][i].get_xaxis().set_ticks([])
        ax[j][i].get_yaxis().set_ticks([])
    fig.colorbar(a, ax=ax.ravel().tolist())
    fig.text(
      0.5,
      0.95,
      "appearance components",
      horizontalalignment="center",
      fontsize=12,
    )
    fig.savefig(
      "images/SAM/SAM-densityModes-meanShape.png",
      pad_inches=0,
      format="png",
      dpi=300,
    )
