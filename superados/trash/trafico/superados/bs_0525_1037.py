
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
# Lane 2
m21 = cv2.imread('./resources/m21.png')
m22 = cv2.imread('./resources/m22.png')
m23 = cv2.imread('./resources/m23.png')
m24 = cv2.imread('./resources/m24.png')
m25 = cv2.imread('./resources/m25.png')

# Area of masks in FE image
a11b = np.sum(np.sum(m11[:,:,1]))/255.
a12b = np.sum(np.sum(m12[:,:,1]))/255.
a13b = np.sum(np.sum(m13[:,:,1]))/255.
a14b = np.sum(np.sum(m14[:,:,1]))/255.
a15b = np.sum(np.sum(m15[:,:,1]))/255.

a21b = np.sum(np.sum(m21[:,:,1]))/255.
a22b = np.sum(np.sum(m22[:,:,1]))/255.
a23b = np.sum(np.sum(m23[:,:,1]))/255.
a24b = np.sum(np.sum(m24[:,:,1]))/255.
a25b = np.sum(np.sum(m25[:,:,1]))/255.

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
rho0 = []

rho11 = []
rho12 = []
rho13 = []
rho14 = []
rho15 = []

rho21 = []
rho22 = []
rho23 = []
rho24 = []
rho25 = []

d112 = []
d113 = []
d114 = []
d115 = []

d212 = []
d213 = []
d214 = []
d215 = []

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

  # FRG in each mask
  frg11 = cv2.bitwise_and(ffrg,m11[:,:,1])
  frg12 = cv2.bitwise_and(ffrg,m12[:,:,1])
  frg13 = cv2.bitwise_and(ffrg,m13[:,:,1])
  frg14 = cv2.bitwise_and(ffrg,m14[:,:,1])
  frg15 = cv2.bitwise_and(ffrg,m15[:,:,1])

  frg21 = cv2.bitwise_and(ffrg,m21[:,:,1])
  frg22 = cv2.bitwise_and(ffrg,m22[:,:,1])
  frg23 = cv2.bitwise_and(ffrg,m23[:,:,1])
  frg24 = cv2.bitwise_and(ffrg,m24[:,:,1])
  frg25 = cv2.bitwise_and(ffrg,m25[:,:,1])

  # Each mask filled pixels
  a11 = np.sum(np.sum(frg11))/255.
  a12 = np.sum(np.sum(frg12))/255.
  a13 = np.sum(np.sum(frg13))/255.
  a14 = np.sum(np.sum(frg14))/255.
  a15 = np.sum(np.sum(frg15))/255.

  a21 = np.sum(np.sum(frg21))/255.
  a22 = np.sum(np.sum(frg22))/255.
  a23 = np.sum(np.sum(frg23))/255.
  a24 = np.sum(np.sum(frg24))/255.
  a25 = np.sum(np.sum(frg25))/255.

  # Each mask traffic density
  rho11.append(a11/a11b)
  rho12.append(a12/a12b)
  rho13.append(a13/a13b)
  rho14.append(a14/a14b)
  rho15.append(a15/a15b)

  rho21.append(a21/a21b)
  rho22.append(a22/a22b)
  rho23.append(a23/a23b)
  rho24.append(a24/a24b)
  rho25.append(a25/a25b)

  # The traffic density on the total mask is the mean of the parts
  rho0.append(np.mean([rho11[-1],rho12[-1],rho13[-1],rho14[-1],rho15[-1],rho21[-1],rho22[-1],rho23[-1],rho24[-1],rho25[-1]]))

  # Delete first value of the list to make room
  if len(rho11) == nplot:
    del(rho11[0])
    del(rho12[0])
    del(rho13[0])
    del(rho14[0])
    del(rho15[0])
    
    del(rho21[0])
    del(rho22[0])
    del(rho23[0])
    del(rho24[0])
    del(rho25[0])
    del(rho0[0])
    
    nupdatei = nupdatei + 1
    
    if nupdatei == nupdate:
      nupdatei = 0
      # Calculate cross correlation
      c112 = np.correlate(rho11[-ncorr:],rho12[-ncorr:],'full')
      c113 = np.correlate(rho11[-ncorr:],rho13[-ncorr:],'full')
      c114 = np.correlate(rho11[-ncorr:],rho14[-ncorr:],'full')
      c115 = np.correlate(rho11[-ncorr:],rho15[-ncorr:],'full')
      
      c121 = np.correlate(rho12[-ncorr:],rho11[-ncorr:],'full')
      c131 = np.correlate(rho13[-ncorr:],rho11[-ncorr:],'full')
      c141 = np.correlate(rho14[-ncorr:],rho11[-ncorr:],'full')
      c151 = np.correlate(rho15[-ncorr:],rho11[-ncorr:],'full')
      
      c212 = np.correlate(rho21[-ncorr:],rho22[-ncorr:],'full')
      c213 = np.correlate(rho21[-ncorr:],rho23[-ncorr:],'full')
      c214 = np.correlate(rho21[-ncorr:],rho24[-ncorr:],'full')
      c215 = np.correlate(rho21[-ncorr:],rho25[-ncorr:],'full')
      
      c221 = np.correlate(rho22[-ncorr:],rho21[-ncorr:],'full')
      c231 = np.correlate(rho23[-ncorr:],rho21[-ncorr:],'full')
      c241 = np.correlate(rho24[-ncorr:],rho21[-ncorr:],'full')
      c251 = np.correlate(rho25[-ncorr:],rho21[-ncorr:],'full')
      
      # Find delay from cross correlation
#      d112.append(np.argmax(c112)-(ncorr-1)) # My version

      d112.append((np.argmax(c112)-np.argmax(c121))/2*tbf)
      d113.append((np.argmax(c113)-np.argmax(c131))/2*tbf)
      d114.append((np.argmax(c114)-np.argmax(c141))/2*tbf)
      d115.append((np.argmax(c115)-np.argmax(c151))/2*tbf)
      
      d212.append((np.argmax(c212)-np.argmax(c221))/2*tbf)
      d213.append((np.argmax(c213)-np.argmax(c231))/2*tbf)
      d214.append((np.argmax(c214)-np.argmax(c241))/2*tbf)
      d215.append((np.argmax(c215)-np.argmax(c251))/2*tbf)
      
      # Plot
      ax0_lines[0][0].set_ydata(rho21)
      ax0_lines[1][0].set_ydata(rho22)
      ax0_lines[2][0].set_ydata(rho23)
      ax0_lines[3][0].set_ydata(rho24)
      ax0_lines[4][0].set_ydata(rho25)
      
      ax1_lines[0].set_ydata(rho0)
      
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
      if len(d213) == nplot:
        del(d212[0])
        del(d213[0])
        del(d214[0])
        del(d215[0])

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

