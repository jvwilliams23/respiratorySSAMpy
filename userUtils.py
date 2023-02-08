import numpy as np
import matplotlib.pyplot as plt
import vedo as v
from pathlib import Path
from skimage import exposure as ex
from sklearn.decomposition import PCA
from skimage import io
from skimage.color import rgb2gray
from skimage import filters
import networkx as nx
from copy import copy
from scipy.spatial.distance import cdist
from pylab import cross,dot,inv

def doPCA(c, expl_var=0.9, quiet=True):
    '''
      Args:
        c = 2D array of components
        quiet: (bool) turn off print checks

      Returns:
        pca, variation of each mode
        k, amount of modes to reach desired variance ratio
    ''' 
    # sc = StandardScaler().fit_transform(c)
    pca = PCA(svd_solver="full")#, n_components = 0.99)
    # pca = SparsePCA(alpha=0.5)
    pca.fit(c)
    varRat = pca.explained_variance_ratio_
    k = np.where(np.cumsum(varRat)>expl_var)[0][0]
    if not quiet:
        print("Reduced to {} components from {} for {}% variation".format(
                                                      k,len(c),expl_var*100)
              )

    return pca, k


def euclideanDist(x, y):
    '''
        Finds the euclidean distance between two arrays x, y.
        Calculated using pythagoras theorem
    '''
    if x.size <= 3:
        return np.sqrt(np.sum((x-y) ** 2))
    else:
        return np.sqrt(np.sum((x-y) ** 2, axis = 1))

def histogram_equalisation(img):
    '''
        Histogram equalisation for contrast enhancement
        https://github.com/AndyHuang1995/Image-Contrast-Enhancement
    '''
    if(len(img.shape)==2):      #gray
        outImg = ex.equalize_hist(img[:,:])*255 
    elif(len(img.shape)==3):    #RGB
        outImg = np.zeros((img.shape[0],img.shape[1],3))
        for channel in range(img.shape[2]):
            outImg[:, :, channel] = ex.equalize_hist(img[:, :, channel])*255

    outImg[outImg>255] = 255
    outImg[outImg<0] = 0
    return outImg.astype(np.uint8)

def loadXR(file):
  '''
    take input X-ray as png (or similar) and convert to grayscale np array

    can add any necessary pre-processing steps in here, 
    such as utils.he (histogram equalisation contrast enhancement) 
  '''
  g_im = io.imread(file, as_gray=True)
  # print(image.shape)
  # g_im = rgb2gray(image)
  g_im = histogram_equalisation(g_im) / 255
  return g_im

def localNormalisation(img, rad=2):
  '''edge preserving filter
    args: img: np.ndarry, image to filtr
        rad: int, radius of filtering kernel
  '''
  g_i = filters.gaussian(img, rad)
  sigma = np.sqrt( filters.gaussian( (img - g_i)**2, rad ) )
  return (img - g_i) / sigma

def mahalanobisDist(x, y):
    '''
        Finds the mahalanobis distance between two arrays x, y
        Calculated based on the inverse covariance matrix of the two arrays
        and the difference of each array (delta)
    '''
    delta = x-y
    if len(np.where(delta == delta[0])[0]) == delta.size:
        return 0
    X = np.vstack([x,y])

    V = np.cov(X.T)

    if 0 in np.diagonal(V):
        print("SINGULAR MATRIX, EXITING")
        exit()
    VI = np.linalg.inv(V)

    if np.sum(np.dot(delta, VI) * delta) < 0:
        return 10000000

    return np.sqrt(np.sum(np.dot(delta, VI) * delta, axis = -1))    

def rotate_coords_about_z(coords, angle=45, origin=[0, 0, 0]):
  """
  Rotate a 3D point cloud around an angle (in degrees).

  Parameters
  ----------
  coords (np.ndarray, N, 3): coordinates in cartesian space
  angle (float or int): angle to rotate the point cloud by
  origin (list or np.array): origin to rotate the points around
  """
  # if coords.ndim == 2:
  x_coords = coords[:, 0] - origin[0]
  y_coords = coords[:, 1] - origin[1]
  rot_x = x_coords*np.cos(np.radians(angle)) - y_coords*np.sin(np.radians(angle))
  rot_y = y_coords*np.cos(np.radians(angle)) + x_coords*np.sin(np.radians(angle))
  rot_coords = np.c_[rot_x + origin[0], rot_y + origin[1], coords[:, 2]]
  return rot_coords

def plotLoss(lossList, scale="linear", wdir="./", tag=""):
  # out_path_obj = Path(wdir)
  # out_path_obj.mkdir(parents=True, exist_ok=True)
  from os import makedirs
  makedirs(wdir, exist_ok=True)
  plt.close()
  fig, ax = plt.subplots(1,1)
  ax.plot(np.arange(0,len(lossList)), lossList, lw=1)
  ax.set_title(tag+" loss function")
  ax.set_yscale("linear")
  ax.set_ylabel("Loss")
  ax.set_xlabel("Iteration")
  for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
  plt.savefig(wdir+"loss"+tag+".pdf")
  return None

def simplifyGraph(G):
  ''' Reduce graph containing all skeletonised voxels to only branch points.
      Loop over the graph until all nodes of degree 2 have been 
      removed and their incident edges fused
      params:
      G: networkx graph
      returns:
      networkx graph 
  '''
  g = G.copy()
  # use while loop as we break the for loop once we remove a mid-point node
  # so loop until there are no nodes with degree = 2 (line to line).
  # We are left with only degree >= 3 (branch) or degree = 1 (end points)
  stopLoop = False # for forcing loop to stop if error found in getting edges
  while any(degree==2 for _, degree in g.degree):
    if stopLoop and sum(degree==2 for _, degree in g.degree)<=1:
      break 
    # prevent error `dictionary changed size during iteration` 
    g0 = g.copy() 
    for node, degree in g.degree():
      # print(g.nodes[node]["pos"])
      if degree==2:
        # for directed graphs we need to maintain direction 
        # (which point is in vs out)
        if g.is_directed(): 
          if len(list(g.in_edges(node))) == 0 or len(list(g.out_edges(node))) == 0:
            # prevent strange issue where no in_edge for some nodes
            stopLoop = True # force while loop to stop
            continue
          a0,b0 = list(g.in_edges(node))[0]
          a1,b1 = list(g.out_edges(node))[0]

        else:
          edges = g.edges(node)
          edges = list(edges)#.__iter__())
          a0,b0 = edges[0]
          a1,b1 = edges[1]

        # decide which nodes to save and which to delete
        if a0 != node:
          e0 = copy(a0)
        else:
          e0 = copy(b0)

        if a1 != node:
          e1 = copy(a1)
        else:
          e1 = copy(b1)

        # remove midpoint and merge two adjacent edges to become one
        g0.remove_node(node)
        g0.add_edge(e0, e1)
        break

    g = g0.copy()
  return g

def trainTestSplit(inputData, train_size=0.9, quiet=False):
    '''
    Args:
    inputData (2D array): (3*num landmarks, num samples)
    train_size (float): 0 < train_size < 1
    quiet (bool) turn off print checks

    Returns:
    train (np.ndarray): training dataset, shape (ntrain, nlandmarks*nfeatures)
    train (np.ndarray): training dataset, shape (ntest, nlandmarks*nfeatures)

    Splits data into training and testing randomly based on a set 
    test size
    '''
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(inputData, 
                                    train_size=train_size)
    if not quiet:
        print("set sizes (train | test)")
        print("\t",train.shape, "|",test.shape)
    return train, test
