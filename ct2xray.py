'''
Convert CT dataset to digitally reconstructed radiograph by summing over
the anterior-posterior direction of the scan

'''
import numpy as np 
import os
import sys
import collections
import pydicom
from pydicom.dicomdir import DicomDir
from pydicom.data import get_testdata_files
import matplotlib.pyplot as plt
# from myshow import myshow
from skimage.measure import label

import SimpleITK as sitk
from sys import exit

import argparse
from datetime import date

debug = False

tag="test"

exact09 = "/home/josh/Dropbox (Heriot-Watt University Team)/"\
         "RES_EPS_Biomechanics/2019-MyLung/06_Research/"\
         "2020-MSc-DBustamante-SAM-Segmentation/imageData/"\
         "exact09/Training/CASE20/"
lidc_dir = "/home/josh/3DSlicer/project/"\
         "LIDC-IDRI/LIDC-IDRI-0002/01-01-2000-98329/3000522.000000-04919/"


def getArgs():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--ctDir', '-i',
                      required=True,
                      type=str, 
                      help='input dir containing ct data'
                      )
  parser.add_argument('--writeDir', '-w',
                      default="./", 
                      type=str, 
                      help='output dir to write x-ray data'
                      )
  parser.add_argument('--out', '-o',
                      default='out.png', 
                      type=str, 
                      help='outline file name (x-ray png)'
                      )
  parser.add_argument('--debug', '-d',
                      default=False, 
                      type=bool, 
                      help='debug mode prints checks and shows image checks'
                      )

  return parser.parse_args()

class ConvertCT2XRay:
  '''
      Class to convert CT dataset to X-ray.
  '''

  def __init__(self, ct_3d, spacing):

    source, destination_board = self.configureRayTracing(ct_3d)
    self.xray_2D = self.ct2xray(source, destination_board, ct_3d)

  def ct2xray(self, xray_source, dest_board, ct_3d):
    '''
    Convert CT dataset to X-ray projection by summing
    Hounsfield units in the anterior-posterior direction.

    Parameters
    ----------
    xray_source (Point3D): source of X-ray camera
    dest_board (List of point3D): corners of X-ray board
    ct_3d (np.ndarray, ndim=3): Hounsfield units of each voxel in CT dataset

    Returns
    -------
    out_scan (np.ndarray, ndim=2): pixel gray-values for new X-ray
    '''

    # Check whether these are simple cases - the board is positioned straight
    # and there are no transitions between the y axis
    same_ys_top = dest_board[0].y == dest_board[1].y
    same_ys_bottom = dest_board[2].y == dest_board[3].y
    same_zs = all(corner.z == dest_board[0].z for corner in dest_board)
    same_ys = all(corner.y == dest_board[0].y for corner in dest_board)
    same_xs = all(corner.x == dest_board[0].x for corner in dest_board)

    out_scan = []
    # iterate over dcm files (y axis in the ct scan)
    board_zs = [corner.z for corner in dest_board]
    # print(board_zs)
    # print("board_ys", board_ys)
    progress_list = [] # check if progress % has been shown
    for z_i, dcm in enumerate(ct_3d[min(board_zs):max(board_zs)]):
      # simplest case - all the corners of the flat are in the same y axis
      # if same_ys:
      assert same_ys, 'corners should have same y coordinate. Others not implemented.'
      row = []
      #-check the direction of the x-ray and 
      #-set the y range for the integral calculation
      if dest_board[0].y > xray_source.y:
        y_range = range(xray_source.y, dest_board[0].y)
      else:
        y_range = reversed(range(xray_source.y + 1, dest_board[0].y - 1))

      # iterate over x axis in the dcm
      board_xs = [corner.x for corner in dest_board]
      for x_i in range(len(dcm[0][min(board_xs):max(board_xs)])):
        new_pixel_val = 0
        # iterate over z axis and find calculate the integral
        for y_i in y_range:
          pixel_val = dcm[y_i][x_i]
          if pixel_val >= 0: # in the cone range
            # calculate the integral
            new_pixel_val += pixel_val

        row.append(new_pixel_val)
      out_scan.append(row)

      #-print percentage until completion as check
      progress = int(z_i/len(ct_3d)*100)
      if progress % 10 == 0 and progress not in progress_list: 
        progress_list.append(progress)
        print(f'\t{int(z_i/len(ct_3d) * 100)} %\r')

    return out_scan

  def configureRayTracing(self, ct):
    '''
    declare source and board points based on CT data size

    Parameters
    ----------
    ct (np.ndarray): CT dataset as numpy array (each value is one voxel)

    Returns
    -------
    source (point3D object): source of X-ray camera (?)
                             coordinate as class object i.e. source.x is x coordinate
    board (list of point3D object): corner coordinates of X-ray board
    '''
    source = Point3D(256, 162, 50)
    board = [Point3D(0, ct[0].shape[1], len(ct)), 
             Point3D(ct[0].shape[0], ct[0].shape[1], len(ct)),
             Point3D(0, ct[0].shape[1], 0), 
             Point3D(ct[0].shape[0], ct[0].shape[1], 0)]
    return source, board


def save_image(data, filename):
  sizes = np.shape(data)     
  fig = plt.figure()
  fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(data, cmap=plt.cm.bone)
  plt.savefig(filename, dpi=500)#plt.cm.bone) 
  plt.close()
  return None

# def show_image(data):
#   # debugging only
#     sizes = np.shape(data)     
#     fig = plt.figure()
#     fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)
#     ax.imshow(data, cmap=plt.cm.bone)
#     plt.show()
#     return None

def truncate(image, min_bound, max_bound):
  '''
  Filter out Hounsfield units below or above a boundary cutoff.

  Parameters
  ----------
  image (np.ndarray): pixel values for any 3D or 2D image
  min_bound (float): lower threshold for filtering out pixel values
  max_bound (float): upper threshold for filtering out pixel values 

  Return
  ------
  image (np.ndarray): truncated pixel values for an image
  '''
  image[image < min_bound] = min_bound
  image[image > max_bound] = max_bound
  return image

def getLargestIsland(segmentation):
  '''
  Take binary segmentation, as sitk.Image or np.ndarray,
  and return largest connected 'island'.
  '''
  if type(segmentation) == sitk.Image:
    seg_sitk = True
    if debug: print("getLargestIsland, changing sitk.Image to array")
    segmentation = sitk.GetArrayFromImage(segmentation)
  
  labels = label(segmentation) #-get connected component
  assert( labels.max() != 0 ) # assume at least 1 connected component
  #-get largest connected region (converts from True/False to 1/0)
  largestIsland = np.array(
                          labels==np.argmax(np.bincount(labels.flat)[1:])+1,
                          dtype=np.int8
                          )
  #-if sitk.Image input, return type sitk.Image
  if seg_sitk:
      largestIsland = sitk.GetImageFromArray(largestIsland)
  return largestIsland

def crop(image):
  """
  Use a connected-threshold estimator to separate background and foreground. 
  Then crop the image using the foreground's axis aligned bounding box.
  Args:
      image (SimpleITK image): An image where the anatomy and background 
                                  intensities form a bi-modal distribution
      seed (list)            : Seed point to grow region from
  Return:
      Cropped image based on foreground's axis aligned bounding box. 
      seed point amended to keep the same relative position.                                
  """
  '''
  Set pixels that are in [min_intensity,otsu_threshold] 
  to inside_value, values above otsu_threshold are set to outside_value. 
  The anatomy has higher intensity values than the background, so it is outside.
  '''
  import itertools
  from itertools import permutations 
  list_1 = [0,0,0]
  list_2 = list(image.GetSize())
  seeds = [[list_1[0], list_1[1], list_1[2]],
           # [list_1[0], list_1[1], list_2[2]],
           # [list_1[0], list_2[1], list_2[2]],
           # [list_2[0], list_2[1], list_2[2]],
           # [list_2[0], list_2[1], list_1[2]],
           # [list_2[0], list_1[1], list_2[2]],
           ]
  # print(seeds)

  lower, upper = -1500, -500 #Hard coded estimates
  im = getLargestIsland(image > -500) #> 50 & image < 2000 #uppe

  label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
  label_shape_filter.Execute( im )
  bounding_box = label_shape_filter.GetBoundingBox(1) #-1 due to binary nature of threshold
  #-The bounding box's first "dim" entries are the starting index and last "dim" entries the size
  roi = sitk.RegionOfInterest(image, 
                              bounding_box[int(len(bounding_box)/2):], 
                              bounding_box[0:int(len(bounding_box)/2)]
                              )

  return roi

def resampleImage(imageIn, scanSize=[512, 512, 512], allAxes = True):
  '''
  Resamples image to improve quality and make isotropically spaced.
  Parameters
  ----------
  imageIn;         SimpleITK Image to be scaled, by a chosen factor.
  scanSize;        Desired output scan size.
  allAxes;         Boolean which determines whether only the width
                   and height are scaled(False) or all 3 axes are 
                   respaced by the spacingFacotr (True).
  '''
  print("Resampling to", scanSize)
  #-Euler transform to get extreme points to resample image
  euler3d = sitk.Euler3DTransform()
  euler3d.SetCenter(imageIn.TransformContinuousIndexToPhysicalPoint(np.array(imageIn.GetSize())/2.0))
  tx = 0
  ty = 0
  tz = 0
  euler3d.SetTranslation((tx, ty, tz))
  extreme_points = [
                  imageIn.TransformIndexToPhysicalPoint((0,0,0)),
                  imageIn.TransformIndexToPhysicalPoint((imageIn.GetWidth(),0,0)),
                  imageIn.TransformIndexToPhysicalPoint((imageIn.GetWidth(),imageIn.GetHeight(),0)),
                  imageIn.TransformIndexToPhysicalPoint((imageIn.GetWidth(),imageIn.GetHeight(),imageIn.GetDepth())),
                  imageIn.TransformIndexToPhysicalPoint((imageIn.GetWidth(),0,imageIn.GetDepth())),
                  imageIn.TransformIndexToPhysicalPoint((0,imageIn.GetHeight(),imageIn.GetDepth())),
                  imageIn.TransformIndexToPhysicalPoint((0,0,imageIn.GetDepth())),
                  imageIn.TransformIndexToPhysicalPoint((0,imageIn.GetHeight(),0))
                  ]
  inv_euler3d = euler3d.GetInverse()
  extreme_points_transformed = [inv_euler3d.TransformPoint(pnt) for pnt in extreme_points]

  min_x = min(extreme_points_transformed)[0]
  min_y = min(extreme_points_transformed, key=lambda p: p[1])[1]
  min_z = min(extreme_points_transformed, key=lambda p: p[2])[2]
  max_x = max(extreme_points_transformed)[0]
  max_y = max(extreme_points_transformed, key=lambda p: p[1])[1]
  max_z = max(extreme_points_transformed, key=lambda p: p[2])[2]

  if allAxes:
    output_spacing = (imageIn.GetSpacing()[0]*imageIn.GetSize()[0]/scanSize[0],
                        imageIn.GetSpacing()[1]*imageIn.GetSize()[1]/scanSize[1],
                        imageIn.GetSpacing()[2]*imageIn.GetSize()[2]/scanSize[2])
  else:
    output_spacing = (imageIn.GetSpacing()[0]*imageIn.GetSize()[0]/scanSize[0],
                        imageIn.GetSpacing()[1]*imageIn.GetSize()[1]/scanSize[1],
                        imageIn.GetSpacing()[2])
  # Identity cosine matrix (arbitrary decision).
  output_direction = imageIn.GetDirection()
  # Minimal x,y coordinates are the new origin.
  output_origin = imageIn.GetOrigin()
  # Compute grid size based on the physical size and spacing.
  # output_size = [int((max_x-min_x)/output_spacing[0]), 
  #                 int((max_y-min_y)/output_spacing[1]), 
  #                 int((max_z-min_z)/output_spacing[2])]
  # for output_axis_size in output_size:
  #   if outpu
  output_size = scanSize

  print("OUTPUT SIZE CHECK", output_size)
  resampled_image = sitk.Resample(imageIn, output_size, euler3d, sitk.sitkLinear, 
                                  output_origin, output_spacing, output_direction)

  return resampled_image

class Point3D(object):
  def __init__(self, x, y, z):
      self.x, self.y, self.z = x, y, z

def dcm2ct3D_sitk(ct_dir):
  '''
  Read some medical image (e.g. dicom stack or mhd), and convert to numpy array

  Parameters
  ----------
  ct_dir (string): /path/to/image.mhd or dicom stack

  Returns
  -------
  image array (np.ndarray, ndim=3): voxel intensity [min value is 0]
  spacing (tuple): voxel spacing - used to get pixel spacing
  '''

  # suppress annoying warnings in reading files
  sitk.ProcessObject_SetGlobalWarningDisplay(False) 
  #-Read image with sitk
  series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(ct_dir)
  if series_IDs: #-check if dicom
    if debug: print("READING DICOM")
    filetype = "DICOM"
    series_file_names = sitk.ImageSeriesReader.\
                            GetGDCMSeriesFileNames(ct_dir, 
                                                    series_IDs[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn() #-For use with data self-obtained
    image3D = series_reader.Execute() #-Get images
  else:
    if debug: print("READING MHD")
    print(ct_dir)
    filetype = "MHD"
    image3D = sitk.ReadImage(ct_dir)
  # now re-activate warnings
  sitk.ProcessObject_SetGlobalWarningDisplay(True)

  #-input to X-ray segmentation CNN requires 500x500 
  newSize = [500,500,500] 
  if debug: print("cropping to size = ", newSize)

  print("Before crop", image3D.GetSpacing(), image3D.GetSize())

  # try to eliminate effect of background objects such as bed
  image3D = crop(image3D)

  print("Before resample", image3D.GetSpacing(), image3D.GetSize())
  image3D = resampleImage(image3D, newSize)
  imArr = sitk.GetArrayFromImage(image3D)
  imArr = truncate(sitk.GetArrayFromImage(image3D),
                   -1024,
                    3071
                   )
  # set min value to 0 so image has minimum gray-value 0 (i.e. air is black on png)
  imArr -= imArr.min()
  print("After resample", image3D.GetSpacing(), image3D.GetSize())

  #-flip output otherwise image appears upside-down 
  # because sitk -> numpy has reversed axes
  return list(np.flip(imArr, axis=0)), image3D.GetSpacing()


if __name__ == '__main__':
  args = getArgs()
  ctDir = args.ctDir
  tag = args.out
  writeDir = args.writeDir
  debug = args.debug

  # if not writeName:
  #   writeName = str(date.today())+"test.csv"
  #   print("No name to write given for X-ray segmentation")
  #   print("Defaulting to", writeName)

  ct, spacing = dcm2ct3D_sitk(ctDir)
  convCTxr = ConvertCT2XRay(ct, spacing)

  # save image and write metadata file with pixel spacing
  save_image(convCTxr.xray_2D, writeDir+"./drr-"+tag)
  spacing = np.array(spacing)#/500*512
  if __name__ == "__main__":
    f = open(writeDir+"./drr-"+tag.replace('.png','.md'), "w")
    f.write("Voxel spacing is\n")
    f.write(str(spacing[0])+" "+str(spacing[1])+" "+str(spacing[2])+"\n")
    f.close()
  # writeDir+"./drr-"+tag+".png", np.array(spacing)
