# This script is just to draw a mask and save it

# Imported libraries
import cv2
import numpy as np
import vca2map

# Define function
vca2map = vca2map.vca2map

# List of points of mask and map
pVCA = []
pMap = []

# Ayuda momentanea
m11 = cv2.imread('/home/damzst/Documents/tpf/video/balkon/m11.png')
m12 = cv2.imread('/home/damzst/Documents/tpf/video/balkon/m12.png')
m13 = cv2.imread('/home/damzst/Documents/tpf/video/balkon/m13.png')
m14 = cv2.imread('/home/damzst/Documents/tpf/video/balkon/m14.png')
m15 = cv2.imread('/home/damzst/Documents/tpf/video/balkon/m15.png')

print m11.shape
# Mouse callback function
def add_point(event,x,y,flags,param):
  global pVCA, pMap
  # Left double-click to add a point to list 
  if event == cv2.EVENT_LBUTTONDBLCLK:
    pVCA.append((x,y))
    # Map coordinate from the vca image to the map
    xm, ym = np.int32(vca2map(x,y))
    pMap.append((xm,ym))
    # Draw lines or not
    if len(pVCA) > 1:
      cv2.line(frame,pVCA[-2],pVCA[-1],(0,255,0),1)
      cv2.line(frMap,pMap[-2],pMap[-1],(0,255,0),1)
    # Draw circles
    cv2.circle(frame,(x,y),1,(0,0,255),-1)
    # Calculate real area
    if len(pVCA)>3:
      areapx2 = cv2.contourArea(np.asarray(pMap))
      aream2 = areapx2 / 0.8**2
      print aream2

# Call mouse function
cv2.namedWindow('Mask definition',1)
cv2.setMouseCallback('Mask definition',add_point)

# Capture desired video
mainFolder = '/home/damzst/Documents/tpf/video'
vidBalkonWinter = '/balkon/balkonWinter_red760.avi'
vidBalkonSummerRed = '/balkon/balkonSummer_red.avi'
vidBalkonSummerBig = '/balkon/balkonSummer.mp4'
chosenOne = vidBalkonSummerRed
cap = cv2.VideoCapture(mainFolder+chosenOne)
# Sometimes there are useless frames at the beginning
uselessframes = 200
cap.set(1,uselessframes)
# Get one frame as reference
refFrame = cap.read()[1]
refFrame = cv2.addWeighted(refFrame,1,m11,0.5,0)
refFrame = cv2.addWeighted(refFrame,1,m13,0.5,0)
refFrame = cv2.addWeighted(refFrame,1,m15,0.5,0)

cap.release()
frame = refFrame.copy()



# Get and image of the map
mapLocation = '/balkon/mapa.png'
refFrMap = cv2.imread(mainFolder+mapLocation)
frMap = refFrMap.copy()

# Create mask image to save all the little masks
mask = np.zeros(frame.shape,np.uint8)
mMap = np.zeros(frMap.shape,np.uint8)

# Main loop
while(1):
  cv2.imshow('Mask definition',frame)
  cv2.imshow('Map',frMap)
  # Keyboard options
  k = cv2.waitKey(1) & 0xFF
  if k == ord('d'):
    # Add little mask to main mask
    lilmask = np.zeros(frame.shape,np.uint8)
    cv2.fillConvexPoly(lilmask,np.array(pVCA),(255,255,255))
    mask = cv2.add(mask,lilmask)
    frame = cv2.addWeighted(refFrame,1,mask,0.5,0)
    pVCA = []
    # Idem but in map
    mapMask = np.zeros(frMap.shape,np.uint8)
    cv2.fillConvexPoly(mapMask,np.array(pMap),(255,255,255))
    mMap = cv2.add(mMap,mapMask)
    frMap = cv2.addWeighted(refFrMap,1,mMap,0.5,0)
    pMap = []
  if k == ord('s'):
    # Save it
    cv2.imwrite(mainFolder+chosenOne[:-4]+'_mask.png',mask)
  elif k == ord('q'):
    break
  elif k == ord('b'):
    del(pVCA[-1])
    del(pMap[-1])
    frMap = refFrMap.copy()
    frame = refFrame.copy()
    for i in range(len(pVCA)):
      # Draw lines or not
      if i > 0:
        cv2.line(frame,pVCA[i-1],pVCA[i],(0,255,0),1)
        cv2.line(frMap,pMap[i-1],pMap[i],(0,255,0),1)
      # Draw circles
      cv2.circle(frame,pVCA[i],1,(0,0,255),-1)

# Destroy windows
cv2.destroyAllWindows()

