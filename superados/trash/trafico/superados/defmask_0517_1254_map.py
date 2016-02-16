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
    pMap.append((x,y))
    # Map coordinate from the Map image to the VCA
    xm, ym = np.int32(mod.map2vca(x,y))
    pVCA.append((xm,ym))
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
vid = './resources/balkonSummer_red760.avi'
cap = cv2.VideoCapture(vid)

# Get rid of first useless frames
uselessframes = 200
cap.set(1,uselessframes)

# Get one frame as reference
refVCA = cap.read()[1]
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
  cv2.imshow('Mask definition',imMap)
  cv2.imshow('VCA',imVCA)
  # Keyboard options
  k = cv2.waitKey(1) & 0xFF
  if k == ord('d'):
    # Add little mask to main mask
    lilmask = np.zeros(imVCA.shape,np.uint8)
    cv2.fillConvexPoly(lilmask,np.array(pVCA),(255,255,255))
    mask = cv2.add(mask,lilmask)
    imVCA = cv2.addWeighted(refMap,1,mask,0.5,0)
    pVCA = []
    # Idem but in map
    mapMask = np.zeros(imMap.shape,np.uint8)
    cv2.fillConvexPoly(mapMask,np.array(pMap),(255,255,255))
    mMap = cv2.add(mMap,mapMask)
    frMap = cv2.addWeighted(refFrMap,1,mMap,0.5,0)
    pMap = []
  if k == ord('s'):
    # Save it
    cv2.imwrite(vid[:-4]+'_mask.png',mask)
  elif k == ord('q'):
    break
  elif k == ord('b'):
    del(pVCA[-1])
    del(pMap[-1])
    imMap = refMap.copy()
    imVCA = refVCA.copy()
    for i in range(len(pVCA)):
      # Draw lines or not
      if i > 0:
        cv2.line(frame,pVCA[i-1],pVCA[i],(0,255,0),1)
        cv2.line(frMap,pMap[i-1],pMap[i],(0,255,0),1)
      # Draw circles
      cv2.circle(frame,pVCA[i],1,(0,0,255),-1)

# Destroy windows
cv2.destroyAllWindows()

