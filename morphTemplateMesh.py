"""
Morph a template surface mesh and landmarks to a new set of landmarks.
This is done using a radial basis function with a gaussian kernel.
Source: Grassi (2011) Med Eng & Phys 

@author: Josh Williams
01/03/21
"""
"""
python morphTemplateMesh.py --plot y \
        --templateMesh /home/josh/3DSlicer/project/luna16Rescaled/case8684/8684_mm_6.stl \
        --templateLMs gamesData/landmarks0RLL-case8684.csv \
        --morphLMs gamesData/landmarksRLL-case9484.csv \
        --case 9484  --sigma 0.3 \
        --groundTruth /home/josh/3DSlicer/project/luna16Rescaled/case9484/9484_mm_6.stl
"""

import vedo as v
import numpy as np
from time import time
from sys import exit
import trimesh
import argparse

class MorphTemplateMesh:

  def __init__(self, lm_template, lm_target, mesh_template, 
                kernel="gaussian", sigma=0.3, c_factor=1, beta=-0.15, 
                smooth=False, debug=False, quiet=False):

    self.sigma = sigma
    alpha = 1
    # scale and align template mesh and landmarks
    coords_template = mesh_template.points()
    coords_template -= coords_template.mean(axis=0) # align mesh to landmarks
    lm_template -= lm_template.mean(axis=0)
    target_offset = lm_target.mean(axis=0)
    lm_target -= target_offset
    size_target = lm_target.max(axis=0) - lm_target.min(axis=0)
    size_template = lm_template.max(axis=0) - lm_template.min(axis=0)
    size_ratio = size_target/size_template
    coords_template *= size_ratio
    lm_template *= size_ratio
    coords_new = coords_template.copy()

    ''' 
    scale to unit standard deviation, such that the gaussian filter is consistent
    with shapes of different sizes
    '''
    std_scale = coords_new.std(axis=0)
    coords_new /= std_scale
    coords_template /= std_scale
    lm_template /= std_scale
    lm_target /= std_scale

    # select functions based on chosen morphing kernel
    if kernel == "gaussian":
      kernel_type = self.gaussian_kernel
      get_weights = self.get_gaussian_weights
    elif kernel == "multiquad":
      self.c = self.euclideanDist(lm_template,lm_target)*c_factor
      kernel_type = self.multiquad_kernel
      get_weights = self.get_multiquad_weights
    elif kernel == "linear":
      kernel_type = self.linear_kernel
      get_weights = self.get_linear_weights

    # morph template surface coordinates
    for i, x in enumerate(coords_template):
      if i%10000==0 and not quiet:
        print("\tPoint", i, "of", coords_template.shape[0])
      w = get_weights(lm_target, lm_template, x)
      kernel = kernel_type(lm_template, x)[:,np.newaxis]
      coords_new[i] = x + np.sum(kernel*w*alpha, axis=0)

      if debug:
        print(kernel, w, np.sum(kernel*w,axis=0))

    # rescale from standard deviation normalised values to real-world
    coords_new -= coords_new.mean(axis=0)
    coords_new *= std_scale
    coords_template *= std_scale
    lm_template *= std_scale
    lm_target *= std_scale
    coords_new += target_offset

    # create mesh object from morphed vertices
    mesh_target = v.Mesh([coords_new, mesh_template.faces()])
    self.mesh_target_base = mesh_target.clone()
    self.coords_template = coords_template
    # smoothing and clean up
    mesh_targettri = mesh_target.to_trimesh()
    watertight = mesh_targettri.is_watertight
    if not quiet: print("Watertight mesh?", watertight)
    if not watertight:
      print("Watertight mesh?", watertight)
      trimesh.repair.fill_holes(mesh_targettri)
      trimesh.repair.broken_faces(mesh_targettri)

    if smooth:
      # trimesh.smoothing.filter_laplacian(mesh_targettri)
      # trimesh.smoothing.dilate_slope(mesh_targettri)
      trimesh.smoothing.filter_humphrey(mesh_targettri, alpha=0.1)
      trimesh.smoothing.filter_humphrey(mesh_targettri, alpha=0.1)

      watertight = mesh_targettri.is_watertight
      if not watertight:
        trimesh.repair.fill_holes(mesh_targettri)
        trimesh.repair.broken_faces(mesh_targettri)
    self.mesh_target = v.trimesh2vedo(mesh_targettri)
    self.coords_new = mesh_target.points()

  def euclideanDist(self, x, y):
      '''
          Finds the euclidean distance between two arrays x, y.
          Calculated using pythagoras theorem
      '''
      if x.size <= 3:
          return np.sqrt(np.sum((x-y) ** 2))
      else:
          return np.sqrt(np.sum((x-y) ** 2, axis = 1))

  def multiquad_kernel(self, lm_template, x):
    """
    Parameters
    -------------
    lm_template: (n, 3) array of coordinates of landmark on template mesh
    x: (3) array of coordinates on mesh to compute distance from template landmarks

    Returns
    -------------
    (n, 1) distances from lm with a kernel function
    """
    return (self.euclideanDist(lm_template,x)**2 + self.c**2)**beta

  def gaussian_kernel(self, lm_template, x):
    """
    Parameters
    -------------
    lm_template: (n, 3) array of coordinates of landmark on template mesh
    x: (3) array of coordinates on mesh to compute distance from template landmarks

    Returns
    -------------
    (n, 1) distances from lm with a kernel function
    """
    return np.exp(-(self.euclideanDist(lm_template, x)**2.)/(2.*self.sigma**2) )

  def linear_kernel(self, lm_template, x):
    """
    Parameters
    -------------
    lm_template: (n, 3) array of coordinates of landmark on template mesh
    x: (3) array of coordinates on mesh to compute distance from template landmarks

    Returns
    -------------
    (n, 1) distances from lm with a 
    """
    return self.euclideanDist(lm_template, x)

  def get_multiquad_weights(self, lm_target, lm_template, x):
    """
    Parameters
    -------------
    lm_target: (n, 3) array of landmark coordinates to morph template mesh to
    lm_template: (n, 3) array of coordinates of landmark in template mesh
    x: (n,3) array of points to compute distance from landmark coord

    Returns
    -------------
    (n, 1) weights to multiply distance kernel at each landmarks
    """
    k = self.multiquad_kernel(lm_template,x)
    w = (lm_target - lm_template)/k.sum()
    return w 

  def get_gaussian_weights(self, lm_target, lm_template, x):
    """
    Parameters
    -------------
    lm_target: (n, 3) array of landmark coordinates to morph template mesh to
    lm_template: (n, 3) array of coordinates of landmark in template mesh
    x: (n,3) array of points to compute distance from landmark coord

    Returns
    -------------
    (n, 1) weights to multiply distance kernel at each landmarks
    """
    k = self.gaussian_kernel(lm_template, x)
    w = (lm_target - lm_template)/k.sum()
    return w 

  def get_linear_weights(self, lm_target, lm_template, x):
    """
    Parameters
    -------------
    lm_target: (n, 3) array of landmark coordinates to morph template mesh to
    lm_template: (n, 3) array of coordinates of landmark in template mesh
    x: (n,3) array of points to compute distance from landmark coord

    Returns
    -------------
    (n, 1) weights to multiply distance kernel at each landmarks
    """
    k = self.linear_kernel(lm_template,x)
    w = (lm_target - lm_template)/k.sum()
    return w 

def inputs():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--templateMesh', 
                      type=str, required=True,
                      help='template mesh'
                      )
  parser.add_argument('--templateLMs', 
                      type=str, required=True,
                      help='template landmarks'
                      )
  parser.add_argument('--morphLMs', 
                      type=str, required=True,
                      help='morph landmarks'
                      )
  parser.add_argument('--case', '--id',
                      type=str, 
                      default="",
                      help='ID for shape'
                      )
  parser.add_argument('--kernel', '-k',
                      type=str, 
                      default="gaussian",
                      help='kernel for morphing'
                      )
  parser.add_argument('--sigma', 
                      type=float, 
                      default=0.3,
                      help='kernel width'
                      )
  parser.add_argument('--debug', '-d',
                      default=False, 
                      type=bool, 
                      help='debug mode -- shows print checks and blocks'+
                          'plotting outputs'
                      )
  parser.add_argument('--plot', 
                      default=False,
                      type=bool, 
                      help='show visualisations of morphing'
                      )
  parser.add_argument('--write', '-w',
                      type=str, 
                      default=False,
                      help='file name to write morphed surface (if none, nothing is written)'
                      )
  parser.add_argument('--groundTruth', '-gt',
                      type=str, 
                      default=False,
                      help='target ground truth for overlaying (not required)'
                      )

  return parser.parse_args()

if __name__ == "__main__":

  args = inputs()
  template_mesh = args.templateMesh
  template_lm = args.templateLMs
  target_lm = args.morphLMs
  kernel = args.kernel
  sigma = args.sigma
  case = args.case
  debug = args.debug
  plot = args.plot
  surf_write = args.write
  ground_truth = args.groundTruth

  key = target_lm.split("landmarks")[1].split("-case")[0]

  st = time() #start time

  '''-------------------------parameters to vary-------------------------'''
  '''
  The c_factor will need adjusted based on the number of landmarks provided.
  c_factor distributes change over large area (compared to landmark distance)
  when its value is large. 
  Small c_factor will create more localised defomations

  beta controls how strongly nodes outside kernel radius are weight 
  (higher beta = higher weights)
  '''
  alpha = 1 
  # beta and c_factor only used with multiquadratic kernel
  beta = -0.1
  c_factor = 1

  #sigma given by inputs() function

  coarsen_nodes = 1 #reads number of vertices/coarsen_nodes as template
  coarsen_lms = 1 #reads number of landmarks/coarsen_lms for adjusting
  smooth = False
  # plot = False
  '''-------------------------parameters to vary-------------------------'''

  # load and pre-process data
  mesh_template = v.load(template_mesh)
  # if coarsen_nodes != 1:
  #   mesh_template.decimate(fraction=1/coarsen_nodes) 
  # coords_template = mesh_template.points()
  # coords_template -= coords_template.mean(axis=0) # align mesh to landmarks
  lm_template = np.loadtxt(template_lm, delimiter=",")[::coarsen_lms]
  lm_target = np.loadtxt(target_lm, delimiter=",")[::coarsen_lms]

  morph = MorphTemplateMesh(lm_template, lm_target, mesh_template,
                            kernel=kernel, sigma=sigma)

  coords_new = morph.coords_new
  coords_template = morph.coords_template

  print("Time taken: ", time()-st)

  # compare template and morphed shapes to evaluate success
  lm_templatev = v.Points(lm_template, r=5)
  lm_targetv = v.Points(lm_target, r=8)
  coords_templatev = v.Points(coords_template, r=2,c="blue")
  coords_newv = v.Points(coords_new, r=2,c="green")
  if plot:
    vp = v.Plotter(N=2, axes=0)
    vp.show(coords_templatev, lm_templatev, at=0)
    vp.show(coords_templatev,coords_newv, lm_targetv, at=1, interactive=True)

  # load and align ground truth mesh
  vol_morphed = morph.mesh_target.volume()
  if ground_truth:
    mesh_true = v.load(ground_truth)
    true_points = mesh_true.points()
    mesh_true = v.Mesh([true_points - true_points.mean(axis=0), 
                        mesh_true.faces()]).alpha(0.3)
    vol_true = mesh_true.volume()
    print("morphed mesh volume {}".format(vol_morphed))
    print("ground truth volume {}".format(vol_true))
    err = abs(vol_true-vol_morphed)/vol_morphed*100
    print("difference = {}%".format(err))

  if plot:
    vp.show(morph.mesh_target_base, at=0)
    if ground_truth:
      vp.show(mesh_true,morph.mesh_target, at=1, interactive=True)
    else:
      vp.show(morph.mesh_target, at=1, interactive=True)

  ''' 
  '''
  if surf_write:
    v.write(morph.mesh_target, surf_write )

