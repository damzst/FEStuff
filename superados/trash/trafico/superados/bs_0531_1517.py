
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

# Get video size
size = cap.get(mod.CV_CAP_PROP_FRAME_WIDTH)

# This Tuesday I looped everything to avoid repetition. It didn't turn up well, so I stepped back. Eventually the loop'll come back.

# Load masks
m = []
lanes = 3
parts = 5
# Main mask
m.append(cv2.imread('./resources/%d/m00.png'%(size),1))
# For each lane and part
for i in range(lanes):
  for j in range(parts):
    m.append(cv2.imread('./resources/%d/m%d%d.png'%(size,i+1,j+1),0))

# Area of masks in VCA image
aVCA11 = np.sum(np.sum(m[1]))/255.
aVCA12 = np.sum(np.sum(m[2]))/255.
aVCA13 = np.sum(np.sum(m[3]))/255.
aVCA14 = np.sum(np.sum(m[4]))/255.
aVCA15 = np.sum(np.sum(m[5]))/255.

aVCA21 = np.sum(np.sum(m[6]))/255.
aVCA22 = np.sum(np.sum(m[7]))/255.
aVCA23 = np.sum(np.sum(m[8]))/255.
aVCA24 = np.sum(np.sum(m[9]))/255.
aVCA25 = np.sum(np.sum(m[10]))/255.

aVCA31 = np.sum(np.sum(m[11]))/255.
aVCA32 = np.sum(np.sum(m[12]))/255.
aVCA33 = np.sum(np.sum(m[13]))/255.
aVCA34 = np.sum(np.sum(m[14]))/255.
aVCA35 = np.sum(np.sum(m[15]))/255.

# Find Contours in Masks
pVCA11 = cv2.findContours(m[1].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
pVCA12 = cv2.findContours(m[2].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
pVCA13 = cv2.findContours(m[3].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
pVCA14 = cv2.findContours(m[4].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
pVCA15 = cv2.findContours(m[5].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]

pVCA21 = cv2.findContours(m[6].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
pVCA22 = cv2.findContours(m[7].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
pVCA23 = cv2.findContours(m[8].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
pVCA24 = cv2.findContours(m[9].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
pVCA25 = cv2.findContours(m[10].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]

pVCA31 = cv2.findContours(m[11].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
pVCA32 = cv2.findContours(m[12].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
pVCA33 = cv2.findContours(m[13].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
pVCA34 = cv2.findContours(m[14].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]
pVCA35 = cv2.findContours(m[15].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0]

# Warp masks into MAP image
pMAP11 = []
for k in range(len(pVCA11)):
  xm, ym = np.int32(mod.vca2map(pVCA11[k,0,0],pVCA11[k,0,1],1920))
  pMAP11.append((xm,ym))
pMAP12 = []
for k in range(len(pVCA12)):
  xm, ym = np.int32(mod.vca2map(pVCA12[k,0,0],pVCA12[k,0,1],1920))
  pMAP12.append((xm,ym))
pMAP13 = []
for k in range(len(pVCA13)):
  xm, ym = np.int32(mod.vca2map(pVCA13[k,0,0],pVCA13[k,0,1],1920))
  pMAP13.append((xm,ym))
pMAP14 = []
for k in range(len(pVCA14)):
  xm, ym = np.int32(mod.vca2map(pVCA14[k,0,0],pVCA14[k,0,1],1920))
  pMAP14.append((xm,ym))
pMAP15 = []
for k in range(len(pVCA15)):
  xm, ym = np.int32(mod.vca2map(pVCA15[k,0,0],pVCA15[k,0,1],1920))
  pMAP15.append((xm,ym))

pMAP21 = []
for k in range(len(pVCA21)):
  xm, ym = np.int32(mod.vca2map(pVCA21[k,0,0],pVCA21[k,0,1],1920))
  pMAP21.append((xm,ym))
pMAP22 = []
for k in range(len(pVCA22)):
  xm, ym = np.int32(mod.vca2map(pVCA22[k,0,0],pVCA22[k,0,1],1920))
  pMAP22.append((xm,ym))
pMAP23 = []
for k in range(len(pVCA23)):
  xm, ym = np.int32(mod.vca2map(pVCA23[k,0,0],pVCA23[k,0,1],1920))
  pMAP23.append((xm,ym))
pMAP24 = []
for k in range(len(pVCA24)):
  xm, ym = np.int32(mod.vca2map(pVCA24[k,0,0],pVCA24[k,0,1],1920))
  pMAP24.append((xm,ym))
pMAP25 = []
for k in range(len(pVCA25)):
  xm, ym = np.int32(mod.vca2map(pVCA25[k,0,0],pVCA25[k,0,1],1920))
  pMAP25.append((xm,ym))

pMAP31 = []
for k in range(len(pVCA31)):
  xm, ym = np.int32(mod.vca2map(pVCA31[k,0,0],pVCA31[k,0,1],1920))
  pMAP31.append((xm,ym))
pMAP32 = []
for k in range(len(pVCA32)):
  xm, ym = np.int32(mod.vca2map(pVCA32[k,0,0],pVCA32[k,0,1],1920))
  pMAP32.append((xm,ym))
pMAP33 = []
for k in range(len(pVCA33)):
  xm, ym = np.int32(mod.vca2map(pVCA33[k,0,0],pVCA33[k,0,1],1920))
  pMAP33.append((xm,ym))
pMAP34 = []
for k in range(len(pVCA34)):
  xm, ym = np.int32(mod.vca2map(pVCA34[k,0,0],pVCA34[k,0,1],1920))
  pMAP34.append((xm,ym))
pMAP35 = []
for k in range(len(pVCA35)):
  xm, ym = np.int32(mod.vca2map(pVCA35[k,0,0],pVCA35[k,0,1],1920))
  pMAP35.append((xm,ym))

# Convert list into an array
pMAP11 = np.asarray(pMAP11)
pMAP12 = np.asarray(pMAP12)
pMAP13 = np.asarray(pMAP13)
pMAP14 = np.asarray(pMAP14)
pMAP15 = np.asarray(pMAP15)

pMAP21 = np.asarray(pMAP21)
pMAP22 = np.asarray(pMAP22)
pMAP23 = np.asarray(pMAP23)
pMAP24 = np.asarray(pMAP24)
pMAP25 = np.asarray(pMAP25)

pMAP31 = np.asarray(pMAP31)
pMAP32 = np.asarray(pMAP32)
pMAP33 = np.asarray(pMAP33)
pMAP34 = np.asarray(pMAP34)
pMAP35 = np.asarray(pMAP35)

# Calculate area in MAP image
aMAP11 = cv2.contourArea(pMAP11) / 0.8**2
aMAP12 = cv2.contourArea(pMAP12) / 0.8**2
aMAP13 = cv2.contourArea(pMAP13) / 0.8**2
aMAP14 = cv2.contourArea(pMAP14) / 0.8**2
aMAP15 = cv2.contourArea(pMAP15) / 0.8**2

aMAP21 = cv2.contourArea(pMAP21) / 0.8**2
aMAP22 = cv2.contourArea(pMAP22) / 0.8**2
aMAP23 = cv2.contourArea(pMAP23) / 0.8**2
aMAP24 = cv2.contourArea(pMAP24) / 0.8**2
aMAP25 = cv2.contourArea(pMAP25) / 0.8**2

aMAP31 = cv2.contourArea(pMAP31) / 0.8**2
aMAP32 = cv2.contourArea(pMAP32) / 0.8**2
aMAP33 = cv2.contourArea(pMAP33) / 0.8**2
aMAP34 = cv2.contourArea(pMAP34) / 0.8**2
aMAP35 = cv2.contourArea(pMAP35) / 0.8**2

# Center of mass of masks im MAP image
M = cv2.moments(pMAP11)
cMAP11 = np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])
M = cv2.moments(pMAP12)
cMAP12 = np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])
M = cv2.moments(pMAP13)
cMAP13 = np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])
M = cv2.moments(pMAP14)
cMAP14 = np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])
M = cv2.moments(pMAP15)
cMAP15 = np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])

M = cv2.moments(pMAP21)
cMAP21 = np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])
M = cv2.moments(pMAP22)
cMAP22 = np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])
M = cv2.moments(pMAP23)
cMAP23 = np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])
M = cv2.moments(pMAP24)
cMAP24 = np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])
M = cv2.moments(pMAP25)
cMAP25 = np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])

M = cv2.moments(pMAP31)
cMAP31 = np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])
M = cv2.moments(pMAP32)
cMAP32 = np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])
M = cv2.moments(pMAP33)
cMAP33 = np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])
M = cv2.moments(pMAP34)
cMAP34 = np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])
M = cv2.moments(pMAP35)
cMAP35 = np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])])

# Find distance between centers of mass.

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

rho31 = []
rho32 = []
rho33 = []
rho34 = []
rho35 = []

d112 = []
d113 = []
d114 = []
d115 = []

d212 = []
d213 = []
d214 = []
d215 = []

d312 = []
d313 = []
d314 = []
d315 = []

# Number of frames between correlations
ncorr = 50
nupdate = 1
nupdatei = 0

# Figure
plt.ion()
fig = plt.figure()

# I have to find a way to show everything, from all masks.

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
tbf = 1/cap.get(mod.CV_CAP_PROP_FPS) # It happens to be constant on the original video. If it weren't, it is easy to calculate but CPU time consuming. At least, that's what I experienced.

# Main Loop
while(cap.isOpened()):
  # Read frame
  fr = cap.read()[1]
  # Apply mask
  fr2 = cv2.bitwise_and(fr,m[0])
  # Apply BKG substractor
  frg = bs.apply(fr2)
  # Filter noise
  blur = cv2.GaussianBlur(frg,(17,17),0)
  ffrg = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

  # FRG in each mask
  frg11 = cv2.bitwise_and(ffrg,m[1])
  frg12 = cv2.bitwise_and(ffrg,m[2])
  frg13 = cv2.bitwise_and(ffrg,m[3])
  frg14 = cv2.bitwise_and(ffrg,m[4])
  frg15 = cv2.bitwise_and(ffrg,m[5])

  frg21 = cv2.bitwise_and(ffrg,m[6])
  frg22 = cv2.bitwise_and(ffrg,m[7])
  frg23 = cv2.bitwise_and(ffrg,m[8])
  frg24 = cv2.bitwise_and(ffrg,m[9])
  frg25 = cv2.bitwise_and(ffrg,m[10])

  frg31 = cv2.bitwise_and(ffrg,m[11])
  frg32 = cv2.bitwise_and(ffrg,m[12])
  frg33 = cv2.bitwise_and(ffrg,m[13])
  frg34 = cv2.bitwise_and(ffrg,m[14])
  frg35 = cv2.bitwise_and(ffrg,m[15])

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

  a31 = np.sum(np.sum(frg31))/255.
  a32 = np.sum(np.sum(frg32))/255.
  a33 = np.sum(np.sum(frg33))/255.
  a34 = np.sum(np.sum(frg34))/255.
  a35 = np.sum(np.sum(frg35))/255.

  # Each mask traffic density
  rho11.append(a11/aVCA11)
  rho12.append(a12/aVCA12)
  rho13.append(a13/aVCA13)
  rho14.append(a14/aVCA14)
  rho15.append(a15/aVCA15)

  rho21.append(a21/aVCA21)
  rho22.append(a22/aVCA22)
  rho23.append(a23/aVCA23)
  rho24.append(a24/aVCA24)
  rho25.append(a25/aVCA25)

  rho31.append(a31/aVCA31)
  rho32.append(a32/aVCA32)
  rho33.append(a33/aVCA33)
  rho34.append(a34/aVCA34)
  rho35.append(a35/aVCA35)

  # The traffic density on the total mask is the mean of the parts
  rho0.append(np.mean([
  rho11[-1],rho12[-1],rho13[-1],rho14[-1],rho15[-1],
  rho21[-1],rho22[-1],rho23[-1],rho24[-1],rho25[-1],
  rho31[-1],rho32[-1],rho33[-1],rho34[-1],rho35[-1]]))

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
    
    del(rho31[0])
    del(rho32[0])
    del(rho33[0])
    del(rho34[0])
    del(rho35[0])
    
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
      
      c312 = np.correlate(rho31[-ncorr:],rho32[-ncorr:],'full')
      c313 = np.correlate(rho31[-ncorr:],rho33[-ncorr:],'full')
      c314 = np.correlate(rho31[-ncorr:],rho34[-ncorr:],'full')
      c315 = np.correlate(rho31[-ncorr:],rho35[-ncorr:],'full')
      
      c321 = np.correlate(rho32[-ncorr:],rho31[-ncorr:],'full')
      c331 = np.correlate(rho33[-ncorr:],rho31[-ncorr:],'full')
      c341 = np.correlate(rho34[-ncorr:],rho31[-ncorr:],'full')
      c351 = np.correlate(rho35[-ncorr:],rho31[-ncorr:],'full')
      
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
      
      d312.append((np.argmax(c312)-np.argmax(c321))/2*tbf)
      d313.append((np.argmax(c313)-np.argmax(c331))/2*tbf)
      d314.append((np.argmax(c314)-np.argmax(c341))/2*tbf)
      d315.append((np.argmax(c315)-np.argmax(c351))/2*tbf)
      
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
        del(d112[0])
        del(d113[0])
        del(d114[0])
        del(d115[0])
        
        del(d212[0])
        del(d213[0])
        del(d214[0])
        del(d215[0])
        
        del(d312[0])
        del(d313[0])
        del(d314[0])
        del(d315[0])

  # Show different steps
  cv2.imshow('1 - Frame',cv2.addWeighted(fr[640:1300,1050:1650],1,cv2.cvtColor(m[15],cv2.COLOR_GRAY2BGR)[640:1300,1050:1650],-1,0))
  cv2.moveWindow('1 - Frame',0,0)
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

