# This script is just to draw a mask and save it

# Imported libraries
import cv2
import numpy as np
import mod

# List of points of mask and map
pVCA = []
pMap = []

# Mouse callback function
def add_point(event,x,y,flags,param):
  global pVCA, pMap
  # Left double-click to add a point to list 
  if event == cv2.EVENT_LBUTTONDBLCLK:
    pVCA.append((x,y))
    # Map coordinate from the vca image to the map
    xm, ym = np.int32(mod.vca2map(x,y))
    print x,y,xm,ym
    pMap.append((xm,ym))
    # Draw lines or not
    if len(pVCA) > 1:
      cv2.line(imVCA,pVCA[-2],pVCA[-1],(0,255,0),1)
      cv2.line(imMap,pMap[-2],pMap[-1],(0,255,0),1)
    # Draw circles
    cv2.circle(imVCA,(x,y),1,(0,0,255),-1)
    # Calculate real area
    if len(pVCA)>3:
      areapx2 = cv2.contourArea(np.asarray(pMap))
      aream2 = areapx2 / 0.8**2
      print aream2

# Call mouse function
cv2.namedWindow('Mask definition',1)
cv2.setMouseCallback('Mask definition',add_point)

# Capture the video
vid = '../../video/balkon/balkonSummer.mp4'
cap = cv2.VideoCapture(vid)

# Get rid of first useless frames
uselessframes = 100
cap.set(1,uselessframes)

# Get one frame as reference
#bbVCA = np.array([[1050,1650],[600,1200]])
#refVCA = cap.read()[1][bbVCA[1,0]:bbVCA[1,1],bbVCA[0,0]:bbVCA[0,1]]
mguia = cv2.add(cv2.add(cv2.add(m11,m13),cv2.add(m15,m21)),cv2.add(m23,m25))
refVCA = cap.read()[1]
refVCA = cv2.addWeighted(refVCA,1,mguia,-1,0)
cap.release()
imVCA = refVCA.copy()

# Get Map image
refMap = cv2.imread('./resources/mapa.png')
imMap = refMap.copy()

# Create mask image to save all the little masks
mask = np.zeros(imVCA.shape,np.uint8)
mMap = np.zeros(imMap.shape,np.uint8)

# Main loop
while(1):
  cv2.imshow('Mask definition',imVCA)
  cv2.imshow('Map',imMap)
  cv2.moveWindow('Mask definition',0,0)
  # Keyboard options
  k = cv2.waitKey(1) & 0xFF
  if k == ord('d'):
    # Add little mask to main mask
    lilmask = np.zeros(imVCA.shape,np.uint8)
    cv2.fillConvexPoly(lilmask,np.array(pVCA),(255,255,255))
    mask = cv2.add(mask,lilmask)
    imVCA0 = imVCA.copy()
    imVCA = cv2.addWeighted(refVCA,1,mask,-1,0)
    pVCA = []
    # Idem but in map
    mapMask = np.zeros(imMap.shape,np.uint8)
    cv2.fillConvexPoly(mapMask,np.array(pMap),(255,255,255))
    mMap = cv2.add(mMap,mapMask)
    imMap0 = imMap.copy()
    imMap = cv2.addWeighted(refMap,0.2,mMap,0.8,0)
    pMap = []
  if k == ord('s'):
    # Save it
    cv2.imwrite(vid[:-4]+'_mask.png',lilmask)
  elif k == ord('q'):
    break
  elif k == ord('b'):
    del(pVCA[-1])
    del(pMap[-1])
    imMap = imMap0.copy()
    imVCA = imVCA0.copy()
    for i in range(len(pVCA)):
      # Draw lines or not
      if i > 0:
        cv2.line(imVCA,pVCA[i-1],pVCA[i],(0,255,0),1)
        cv2.line(imMap,pMap[i-1],pMap[i],(0,255,0),1)
      # Draw circles
      cv2.circle(imVCA,pVCA[i],1,(0,0,255),-1)
      cv2.circle(imMap,pMap[i],1,(0,0,255),-1)

# Destroy windows
cv2.destroyAllWindows()

