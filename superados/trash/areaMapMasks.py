# Script to compare our simple algorithm to those in the library

# Imported libraries
import numpy as np
import cv2
import vca2map

# Define functions
vca2map = vca2map.vca2map

def areaMap(m):
  # Calculate the area in MAP image of the entry mask
  aux = m[:,:,1].copy()
  c = cv2.findContours(aux,cv2.RETR_EXTERNAL,2)[1]
  # Load list as array and remove 1-dim entries
  ca = np.squeeze(c)
  pMap = []
  for i in range(len(ca)):
    xm, ym = np.int32(vca2map(ca[i,0],ca[i,1]))
    pMap.append((xm,ym))
  areapx2 = cv2.contourArea(np.asarray(pMap))
  aream2 = areapx2 / 0.8**2
  return aream2

# Load mask
mainFolder = '/home/damzst/Documents/tpf/video'
mask = cv2.imread(mainFolder+'/balkon/m0.png')

# Masks
# Lane 1
m11 = cv2.imread(mainFolder+'/balkon/m11.png')
m12 = cv2.imread(mainFolder+'/balkon/m12.png')
m13 = cv2.imread(mainFolder+'/balkon/m13.png')
m14 = cv2.imread(mainFolder+'/balkon/m14.png')
m15 = cv2.imread(mainFolder+'/balkon/m15.png')

# Mask area in FE image
a11fe = np.sum(np.sum(m11[:,:,1]))/255.
a12fe = np.sum(np.sum(m12[:,:,1]))/255.
a13fe = np.sum(np.sum(m13[:,:,1]))/255.
a14fe = np.sum(np.sum(m14[:,:,1]))/255.
a15fe = np.sum(np.sum(m15[:,:,1]))/255.

# Mask area in MAP image
a11m = areaMap(m11)
a12m = areaMap(m12)
a13m = areaMap(m13)
a14m = areaMap(m14)
a15m = areaMap(m15)

print a11m, a12m, a13m, a14m, a15m
