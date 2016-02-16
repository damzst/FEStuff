
# Imported libraries
from matplotlib import pyplot as plt
import numpy as np
import cv2
import mod

# Capture the video
cap = cv2.VideoCapture('../../video/balkon/balkonSummer.mp4')

# Get rid of first useless frames
uselessframes = 200
cap.set(1,uselessframes)

# Load all masks and properties
m = []
aVCA = np.zeros((3,5))
aMAP = np.zeros((3,5))
cMAP = np.zeros((3,5,2))
for i in range(3):
  for j in range(5):
    # Load mask
    m.append(cv2.imread('./resources/1920/m'+str(i+1)+str(j+1)+'.png'))
    mij = m[-1][:,:,1].copy()
    # Calculate area in VCA image
    aVCA[i][j] = np.sum(np.sum(mij))/255.
    # Find Contours in Masks
    pVCA = cv2.findContours(mij,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    # Warp masks into MAP image
    pMAP = []
    for k in range(len(pVCA)):
      x = pVCA[k][0]
      y = pVCA[k][1]
      xm, ym = np.int32(mod.vca2map(x,y))
      pMAP.append((xm,ym))
    # Calculate area in MAP image
    aMAP[i,j] = cv2.contourArea(np.asarray(pMAP)) / 0.8**2
    # Center of mass of masks im MAP image
    M = cv2.moments(cnt)
    cMAP[i,j,0] = int(M['m10']/M['m00'])
    cMAP[i,j,1] = int(M['m01']/M['m00'])


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
tms = 10

# Number of frames to plot
nplot = 50

# List of samples:
RHO0 = []
RHO = []
DELAY = []

# Number of frames between correlations
ncorr = 50
nupdate = 1
nupdatei = 0

# Figure
plt.ion()
fig = plt.figure()

# Subplot 0: Density by parts
ax0 = fig.add_subplot(311)
ax0.axis([0,nplot,0,1])
ax0.set_ylabel('Densidad por tramo [%]')
ax0_color = ['r','g','b','c','m']
ax0_label = ['rho11','rho12','rho13','rho14','rho15']
ax0_lines = []
ax0_xdata = np.arange(nplot-1)
ax0_ydata = np.zeros(nplot-1)
for i in range(len(ax0_label)):
  ax0_lines.append(ax0.plot(ax0_xdata,ax0_ydata,ax0_color[i],label=ax0_label[i]))

# Subplot 1: Total density
ax1 = fig.add_subplot(312)
ax1.axis([0,nplot,0,1])
ax1.set_ylabel('Densidad total [%]')
ax1_lines = ax1.plot(ax0_ydata,ax0_color[0],label='rho')

# Subplot 2: Delays in secs
ax2 = fig.add_subplot(313)
ax2.axis([0,nplot,0,500])
ax2.set_ylabel('Delay [s]')
ax2_label = ['d12','d13','d14','d15']
ax2_lines = []
for i in range(len(ax2_label)):
  ax2_lines.append(ax2.plot(ax0_ydata,ax0_color[i],label=ax2_label[i]))

# Time between frames in seg
tbf = 1/cap.get(mod.CV_CAP_PROP_FPS)

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

  # Main For
  aFRG = np.zeros((3,5))
  RHOn = np.zeros((3,5))
  FRGn = []
  for i in range(3):
    for j in range(5):
      # FRG in each mask
      FRGn.append(cv2.bitwise_and(ffrg,m[i+3*j][:,:,1]))
      # Each mask filled pixels
      aFRG[i,j] = np.sum(np.sum(FRGn[-1]))/255.
      # Each mask traffic density
      RHOn[i,j] = aFRG[i,j]/aVCA[i,j]
  RHO.append(RHOn)
  
  # Mean traffic density
  RHO0.append(np.mean([RHOn]))

  # Delete first value of the list to make room
  if len(RHO) == nplot:
    del(RHO[0])
    del(RHO0[0])
    
    nupdatei = nupdatei + 1
    if nupdatei == nupdate:
      nupdatei = 0
      npRHO = np.asarray(RHO)
      c = np.zeros((3,4,2,2*nplot-1))
      dij = np.zeros((3,4))
      for i in range(3):
        for j in range(4):
          # Calculate cross correlations
          c[i,j,0] = np.correlate(npRHO[-ncorr:,i,1],npRHO[-ncorr:,i,j+1],'full')
          c[i,j,1] = np.correlate(npRHO[-ncorr:,i,j+1],npRHO[-ncorr:,i,1],'full')
          # Find delays from cross correlations
          dij[i,j] = (np.argmax(c[i,j,0])-np.argmax(c[i,j,1]))/2*tbf
      d.append(dij)
      
      # Plot
      ax0_lines[0][0].set_ydata(npRHO[:,0,0])
      ax0_lines[1][0].set_ydata(npRHO[:,1,0])
      ax0_lines[2][0].set_ydata(npRHO[:,2,0])
      ax0_lines[3][0].set_ydata(npRHO[:,3,0])
      ax0_lines[4][0].set_ydata(npRHO[:,4,0])
      
      ax1_lines[0].set_ydata(RHO0)
      
      for i in range(4):
      ax2_xdata = range(len(d212))
      ax2_lines[0][0].set_xdata(ax2_xdata)
      ax2_lines[1][0].set_xdata(ax2_xdata)
      ax2_lines[2][0].set_xdata(ax2_xdata)
      ax2_lines[3][0].set_xdata(ax2_xdata)
      
      ax2_lines[0][0].set_ydata(d212)
      ax2_lines[1][0].set_ydata(d213)
      ax2_lines[2][0].set_ydata(d214)
      ax2_lines[3][0].set_ydata(d215)
      
      plt.draw()
      if len(d) == nplot:
        del(d[0])

  # Show different steps
  cv2.imshow('1 - Frame',fr)

  # Playback buttons
  k = cv2.waitKey(tms) & 0xFF
  if k == ord('q'):
    break # Quit
  elif k == ord('p'):
    tms = 0 #Pause
  elif k == ord('f'):
    tms = 10 # Play

# Release everything
cap.release()
cv2.destroyAllWindows()

