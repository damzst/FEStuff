# Script for... I'll tell you later... when I get it

# Imported libraries
import numpy as np
import cv2
import mod
from matplotlib import pyplot as plt

# Open file
f = open('rho1xs.txt','w')

# Define function
vca2map = mod.vca2map

# Capture the chosen video
mainFolder = '/home/damzst/Documents/tpf/video'
vidBalkonSummerRed = '/balkon/balkonSummer_red.avi'
chosenOne = vidBalkonSummerRed
cap = cv2.VideoCapture(mainFolder+chosenOne)

# Get rid of first useless frames
uselessframes = 200
cap.set(1,uselessframes)

print 'Frames en total: ',cap.get(mod.CV_CAP_PROP_FRAME_COUNT)
print 'FPS: ',cap.get(mod.CV_CAP_PROP_FPS)

# Load mask
mask = cv2.imread(mainFolder+'/balkon/m0.png')

# Masks
# Lane 1
m11 = cv2.imread(mainFolder+'/balkon/m11.png')
m12 = cv2.imread(mainFolder+'/balkon/m12.png')
m13 = cv2.imread(mainFolder+'/balkon/m13.png')
m14 = cv2.imread(mainFolder+'/balkon/m14.png')
m15 = cv2.imread(mainFolder+'/balkon/m15.png')

# Mask area in FE image
a11b = np.sum(np.sum(m11[:,:,1]))/255.
a12b = np.sum(np.sum(m12[:,:,1]))/255.
a13b = np.sum(np.sum(m13[:,:,1]))/255.
a14b = np.sum(np.sum(m14[:,:,1]))/255.
a15b = np.sum(np.sum(m15[:,:,1]))/255.

# Create the BKG substractor
bs = cv2.createBackgroundSubtractorMOG2()
bs.setDetectShadows(True)
bs.setShadowValue(0)
bs.setHistory(2000)
bs.setVarThreshold(36)
bs.setBackgroundRatio(0.5)

# Map location
mapLocation = '/balkon/mapa.png'

# Time in ms to wait between frames, 0 = forever
tfr = 0

# Number of frames to plot
nplot = 200

# Vectors of samples:
rho11 = []
rho12 = []
rho13 = []
rho14 = []
rho15 = []

# Number of frames between correlations
ncorr = 25
ncorri = ncorr

# Number of plot
#pltn = 0

# Main Loop
while(cap.isOpened()):
  # Read frame  
  fr = cap.read()[1]
  # Apply mask
  fr2 = cv2.bitwise_and(fr,mask)
  # Apply BKG substractor
  frg = bs.apply(fr2)

  # Filter noise
  blur = cv2.GaussianBlur(frg,(17,17),0)
  ffrg = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

  # FRG in each mask
  frg11 = cv2.bitwise_and(ffrg,m11[:,:,1])
  frg12 = cv2.bitwise_and(ffrg,m12[:,:,1])
  frg13 = cv2.bitwise_and(ffrg,m13[:,:,1])
  frg14 = cv2.bitwise_and(ffrg,m14[:,:,1])
  frg15 = cv2.bitwise_and(ffrg,m15[:,:,1])

  if len(rho11) == nplot:
    del(rho11[0])
    del(rho12[0])
    del(rho13[0])
    del(rho14[0])
    del(rho15[0])

    if ncorri == ncorr:
      ncorri = 0
      # Calculate cross correlation to find delay
      c112 = np.correlate(rho11[-ncorr:],rho12[-ncorr:],'full')
      d112 = np.argmax(c112)-(ncorr-1)
      d113 = np.argmax(np.correlate(rho11[-25:],rho13[-25:],'full'))-(ncorr-1)
      d114 = np.argmax(np.correlate(rho11[-25:],rho14[-25:],'full'))-(ncorr-1)
      d115 = np.argmax(np.correlate(rho11[-25:],rho15[-25:],'full'))-(ncorr-1)
      # Plot
      plt.close()
      plt.axis([0,200,0,1])
      plt.plot(np.arange(0,len(rho11)),rho11,'-b',label='rho11')
#      plt.hold(True)
      plt.plot(np.arange(0,len(rho12)),rho12,'-g',label='rho12 '+str(d112))
      plt.plot(np.arange(0,len(rho13)),rho13,'-r',label='rho13 '+str(d113))
      plt.plot(np.arange(0,len(rho14)),rho14,'-c',label='rho14 '+str(d114))
      plt.plot(np.arange(0,len(rho15)),rho15,'-m',label='rho15 '+str(d115))
      plt.legend(loc=2)
      plt.title("Vectores de rho")
#      plt.savefig("pltRhos"+str(pltn)+".png")
#      plt.hold(False)
#      pltn = pltn + 1
      plt.show(block=True)
      

    ncorri = ncorri + 1

  a11 = np.sum(np.sum(frg11))/255.
  a12 = np.sum(np.sum(frg12))/255.
  a13 = np.sum(np.sum(frg13))/255.
  a14 = np.sum(np.sum(frg14))/255.
  a15 = np.sum(np.sum(frg15))/255.

  rho11.append(a11/a11b)
  rho12.append(a12/a12b)
  rho13.append(a13/a13b)
  rho14.append(a14/a14b)
  rho15.append(a15/a15b)

#  rho11s = "%s" %(a11/a11b)
#  rho12s = "%s" %(a12/a12b)
#  rho13s = "%s" %(a13/a13b)
#  rho14s = "%s" %(a14/a14b)
#  rho15s = "%s" %(a15/a15b)

#  f.write(rho11s+' '+rho12s+' '+rho13s+' '+rho14s+' '+rho15s+'\n')

  # Find contours

  # Draw preliminary contours
#  cv2.drawContours(fr,cnt,-1,(0,255,0),4)

  # Create map image
#  frm = cv2.imread(mainFolder+mapLocation)

  # Draw desired contours
#  cv2.drawContours(fr,dcnt,-1,(0,0,255),2)

  # Show different steps
  cv2.imshow('1 - Frame',fr)
#  cv2.imshow('2 - Map',cv2.flip(frm,1))
#  cv2.imshow('2 - Foreground',frg)
#  cv2.imshow('3 - Filtered Foreground',ffrg)

  # Keep pressing any key to continue processing
  # Press 'q' to stop
  k = cv2.waitKey(tfr) & 0xFF
  if k == ord('q'):
    break
  elif k == ord('p'):
    tfr = 0
  elif k == ord('f'):
    tfr = 10

# Release everything
cap.release()
cv2.destroyAllWindows()
#f.close()


