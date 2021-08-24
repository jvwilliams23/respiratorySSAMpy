'''
Extract weak outline from X-rays based on conventional image processing methods

'''
# general libs
import numpy as np 
import matplotlib.pyplot as plt
import os 
from os import path
from glob import glob  
from sys import exit
import argparse

# image processing libs and functions
from skimage import filters, io, feature, util, morphology
from skimage.measure import label, regionprops, find_contours, \
                            approximate_polygon
from skimage.morphology import disk, area_closing
from skimage.color import rgb2gray
from skimage import exposure as ex

from skimage.restoration import denoise_bilateral
from skimage.filters.rank import mean_bilateral
from skimage.filters import rank
from skimage.morphology import ball

def getArgs():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--inputFile', '-i',
                      required=True,
                      type=str, 
                      help='input file containing X-ray image as png'
                      )
  parser.add_argument('--outlineFileName', '-o',
                      default='drr-outline.csv', 
                      type=str, 
                      help='file name for in same directory as input file'
                      )
  parser.add_argument('--debug', '-d',
                      default=False, 
                      type=str, 
                      help='debug mode prints checks and shows image checks'
                      )

  return parser.parse_args()

def he(img):
    '''
        Histogram equalisation for contrast enhancement
        https://github.com/AndyHuang1995/Image-Contrast-Enhancement
    '''
    # print("HE shape check", len(img.shape))
    # print(img.shape)
    if(len(img.shape)==2):      #gray
        outImg = ex.equalize_hist(img[:,:])*255 
    elif(len(img.shape)==3):    #RGB
        outImg = np.zeros((img.shape[0],img.shape[1],3))
        for channel in range(img.shape[2]):
            outImg[:, :, channel] = ex.equalize_hist(img[:, :, channel])*255

    outImg[outImg>255] = 255
    outImg[outImg<0] = 0
    return outImg.astype(np.uint8)

def loadXR(file, c=1):
  '''load grayscale image
  '''
  fig = rgb2gray(io.imread(file))
  fig[0] = fig[1].copy()
  return fig[::c,::c]

def filterXR(img, gkernel=2):
  '''edge preserving filter
    args: img: np.ndarry, image to filtr
        gkernel: int, radius of gaussian kernel for filtering
  '''
  # imgE = bilateralfilter(img, gkernel)
  imgE = he(img)
  g = filters.gaussian(imgE, (gkernel,gkernel),2)
  imgE = he(imgE-g)
  return imgE

def bilateralfilter(img, rad=2):
  '''edge preserving filter
    args: img: np.ndarry, image to filtr
        rad: int, radius of filtering kernel
  '''
  return mean_bilateral(img, disk(rad))

def localNormalisation(img, rad=2):
  '''edge preserving filter
    args: img: np.ndarry, image to filtr
        rad: int, radius of filtering kernel
  '''
  g_i = filters.gaussian(img, rad)
  sigma = np.sqrt( filters.gaussian( (img - g_i)**2, rad ) )
  return (img - g_i) / sigma


def getOutline(img, sigma=2):
  '''get outline of edges based on image gradient (canny edges)
  '''
  imEdge = feature.canny(img, sigma=sigma)
  return imEdge 

def compareOutline(img_orig, img_outline, outDir="./", tag="a.png"):
  '''overlays edge map on input image
  '''
  plt.imshow(img_orig, cmap=plt.cm.bone) 
  plt.imshow(img_outline, alpha=0.2)
  # plt.show()
  plt.savefig(outDir+tag,dpi=200)
  return None

def outlineCoords(img):
  '''
    Get outline of lungs from an edge map
  '''
  contours =  find_contours(img, 0.5)
  return contours

# gkernelList = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35]
# gkernelList = [1,2,3,4,5,6,7,8]
from sklearn.preprocessing import MinMaxScaler
if __name__ == "__main__":
  args = getArgs()
  coarsener = 4
  patientID = os.path.basename(os.path.dirname(args.inputFile))
  spacingFile = args.inputFile.replace(".png", ".md")
  spacing =  np.loadtxt(spacingFile, skiprows=1)

  #-load x-ray and pre-process --------------INSERT AIRWAY ENHANCE HERE
  img = loadXR(args.inputFile, coarsener)
  img_filt = img.copy()
  img_filt = he(img)
  # img_filt = filterXR(img, 10)
  # normalise image 
  if img_filt.max()>1:
    img_filt = img_filt/img_filt.max()

  #-get image with outlines 
  out = getOutline(img_filt)
  #-write comparison figure for checking quality of edge map
  # compareOutline(img_filt, out, compareDir, compNameOut)

  #-get outline and write to txt file
  coords = np.vstack(outlineCoords(out))
  #-format edge map for output
  coords = coords[:,[1,0]]
  coords[:,1] *= -1
  coords *= coarsener
  coords -= np.array([250,-250])
  coords *= spacing[[0,2]]

  extent=[-img.shape[1]/2.*spacing[0]*coarsener,  
          img.shape[1]/2.*spacing[0]*coarsener,  
          -img.shape[0]/2.*spacing[2]*coarsener,  
          img.shape[0]/2.*spacing[2]*coarsener ]

  if args.debug:
    plt.close()
    plt.imshow(img, cmap=plt.cm.bone, extent=extent)
    plt.scatter(coords[:,0], coords[:,1], s=5, c='black')
    if args.debug == 's':
      plt.savefig('outline_compare/'+args.inputFile.split('/')[-1])
    else:
      plt.show()
  #-write coordinates
  out_dir = path.dirname(args.inputFile)
  out_name = path.join(out_dir, args.outlineFileName)
  # imDir = "DRRs_edgeMapWorkdir/luna16/{}/".format(patientID)
  # np.savetxt(compareDir+outlineCompareNameOut, coords, delimiter=",")
  np.savetxt(out_name, coords, delimiter=",")
  # exit()
