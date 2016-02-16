# Script for... I'll tell you later... when I get it

# Imported libraries
from matplotlib import pyplot as plt
import numpy as np
import cv2
import mod

# Capture the video
cap = cv2.VideoCapture('./resources/balkonSummer_red760.avi')

# Get rid of first useless frames
uselessframes = 200
cap.set(1,uselessframes)

# Load mask
mask = cv2.imread('./resources/m0.png')

# Masks
# Lane 1
m11 = cv2.imread('./resources/m11.png')
m12 = cv2.imread('./resources/m12.png')
m13 = cv2.imread('./resources/m13.png')
m14 = cv2.imread('./resources/m14.png')
m15 = cv2.imread('./resources/m15.png')

# Area of masks in FE image
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
mapLocation = './resources/mapa.png'

# Time in ms to wait between frames, 0 = forever
tfr = 0

# Number of frames to plot
nplot = 200

# List of samples:
rho11 = []
rho12 = []
rho13 = []
rho14 = []
rho15 = []

# Number of frames between correlations
ncorr = 25
nupdate = 5
nupdatei = 0

# Figure
plt.axis([0,nplot,0,1])
plt.ion()
plt.legend(loc=2)
plt.title("Vectores de rho")
plt.show()
ln_color = ['r','g','b','c','m']
ln_label = ['rho11','rho12','rho13','rho14','rho15']
ln_lines = []
ln_xdata = np.arange(nplot-1)
ln_ydata = np.zeros(nplot-1)
for ii in range(5):
  ln_lines.append(plt.plot(ln_xdata,ln_ydata,ln_color[ii],label=ln_label[ii]))

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

    if nupdatei == nupdate:
      nupdatei = 0
      # Calculate cross correlation
      c112 = np.correlate(rho11[-ncorr:],rho12[-ncorr:],'full')
      c113 = np.correlate(rho11[-ncorr:],rho13[-ncorr:],'full')
      c114 = np.correlate(rho11[-ncorr:],rho14[-ncorr:],'full')
      c115 = np.correlate(rho11[-ncorr:],rho15[-ncorr:],'full')
      # Find delay from cross correlation
      d112 = np.argmax(c112)-(ncorr-1)
      d113 = np.argmax(c113)-(ncorr-1)
      d114 = np.argmax(c114)-(ncorr-1)
      d115 = np.argmax(c115)-(ncorr-1)
      # Plot
      ln_lines[0][0].set_ydata(rho11)
      ln_lines[1][0].set_ydata(rho12)
      ln_lines[2][0].set_ydata(rho13)
      ln_lines[3][0].set_ydata(rho14)
      ln_lines[4][0].set_ydata(rho15)
      plt.draw()
    nupdatei = nupdatei + 1

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

  # Show different steps
  cv2.imshow('1 - Frame',fr)

  # Playback buttons
  k = cv2.waitKey(tfr) & 0xFF
  if k == ord('q'):
    break # Quit
  elif k == ord('p'):
    tfr = 0 #Pause
  elif k == ord('f'):
    tfr = 10 # Play

# Release everything
cap.release()
cv2.destroyAllWindows()

