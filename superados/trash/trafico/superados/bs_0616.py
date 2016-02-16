
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
# Set desired ROI
u0,u1,v0,v1 = 640,1300,1050,1650

# Define px/m ratio
px_m = 0.8

# Lists
m = [] # Masks
aVCA = [] # Mask areas in VCA image
aMAP = [] # Mask areas in MAP image
pVCA = [] # Mask contours in VCA image
pMAP = [] # Mask contours in VCA image
cMAP = [] # Centers of mass in VCA image
# Load masks and calculate related lists
lanes,parts = 3,5
# Main mask
m00 = cv2.imread('./resources/%d/m00.png'%(size),1)[u0:u1,v0:v1]
# For each lane and part
for i in range(lanes):
  for j in range(parts):
    # Load masks
    m.append(cv2.imread('./resources/%d/m%d%d.png'%(size,i+1,j+1),0)[u0:u1,v0:v1])
    # Area of masks in VCA image
    aVCA.append(np.sum(np.sum(m[-1]))/255.)
    # Find contours in masks
    pVCA.append(cv2.findContours(m[-1].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1][0])
    # Warp masks into MAP image
    pMAPij = []
    for k in range(len(pVCA[-1])):
      xm, ym = np.int32(mod.vca2map(pVCA[-1][k,0,0],pVCA[-1][k,0,1],size))
      pMAPij.append((xm,ym))
    # Convert list into an array and append
    pMAP.append(np.asarray(pMAPij))
    # Calculate area in MAP image
    aMAP.append(cv2.contourArea(pMAP[-1])/px_m**2)
    # Center of mass of masks im MAP image
    M = cv2.moments(pMAP[-1])
    cMAP.append(np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])]))

# Calculate distances between centers of mass of masks
dMAP = [] # Distances in m
# For each lane and part but the first
for i in range(lanes):
  for j in range(1,parts):
    dMAP.append(np.sqrt((cMAP[i*parts][0]-cMAP[i*parts+j][0])^2+(cMAP[i*parts][1]-cMAP[i*parts+j][1])^2)/px_m)

# Create the BKG substractor
bs = cv2.createBackgroundSubtractorMOG2()
bs.setDetectShadows(True)
bs.setShadowValue(0)
bs.setHistory(2000)
bs.setVarThreshold(36)
bs.setBackgroundRatio(0.5)

# Map location
mapLocation = './resources/map.png'

# Time in ms to wait between frames, 0 = forever
tms = 10

# Number of frames to plot
nplot = 100

# List of samples:
rho00 = []

rho11,rho12,rho13,rho14,rho15 = ([] for i in range(5))
rho21,rho22,rho23,rho24,rho25 = ([] for i in range(5))
rho31,rho32,rho33,rho34,rho35 = ([] for i in range(5))


s112,s113,s114,s115 = ([] for i in range(4))
s212,s213,s214,s215 = ([] for i in range(4))
s312,s313,s314,s315 = ([] for i in range(4))

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

# Subplot 1: Correlation
ax1 = fig.add_subplot(312)
ax1.axis([0,nplot,-0.1,1.2])
ax1.set_ylabel('Correlacion')
ax1_lines = ax1.plot(ax0_xdata,ax0_ydata,'r')

# Subplot 2: Speed in km/h
ax2 = fig.add_subplot(313)
ax2.axis([0,nplot,0,200])
ax2.set_ylabel('Velocidad [km/h]')
ax2_lines = ax2.plot(ax0_ydata,'r',label='s212')

# Info from capture
fps = 1./cap.get(mod.CV_CAP_PROP_FPS) # Time between frames in seg, it happens to be constant on the original video. If it weren't, it is easy to calculate but CPU time consuming. At least, that's what I experienced.
tot = cap.get(mod.CV_CAP_PROP_FRAME_COUNT) # Frame count

# Font for text in image
font = cv2.FONT_HERSHEY_SIMPLEX

# Draw Contours in another image
mcol = np.zeros(m00.shape,np.uint8)

cv2.fillConvexPoly(mcol,np.array(pVCA[0]),(0,0,255))
cv2.fillConvexPoly(mcol,np.array(pVCA[1]),(0,255,0))
cv2.fillConvexPoly(mcol,np.array(pVCA[2]),(255,0,0))
cv2.fillConvexPoly(mcol,np.array(pVCA[3]),(255,255,0))
cv2.fillConvexPoly(mcol,np.array(pVCA[4]),(255,0,255))

cv2.fillConvexPoly(mcol,np.array(pVCA[5]),(0,0,150))
cv2.fillConvexPoly(mcol,np.array(pVCA[6]),(0,150,0))
cv2.fillConvexPoly(mcol,np.array(pVCA[7]),(150,0,0))
cv2.fillConvexPoly(mcol,np.array(pVCA[8]),(150,150,0))
cv2.fillConvexPoly(mcol,np.array(pVCA[9]),(150,0,150))

cv2.fillConvexPoly(mcol,np.array(pVCA[10]),(0,0,90))
cv2.fillConvexPoly(mcol,np.array(pVCA[11]),(0,90,0))
cv2.fillConvexPoly(mcol,np.array(pVCA[12]),(90,0,0))
cv2.fillConvexPoly(mcol,np.array(pVCA[13]),(90,90,0))
cv2.fillConvexPoly(mcol,np.array(pVCA[14]),(90,0,90))

# Main Loop
while(cap.isOpened()):
  # Read frame
  fr = cap.read()[1][u0:u1,v0:v1]
  # Apply mask
  fr2 = cv2.bitwise_and(fr,m00)
  # Apply BKG substractor
  frg = bs.apply(fr2)
  # Filter noise
  blur = cv2.GaussianBlur(frg,(17,17),0)
  ffrg = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

  # FRG in each mask
  frg11 = cv2.bitwise_and(ffrg,m[0])
  frg12 = cv2.bitwise_and(ffrg,m[1])
  frg13 = cv2.bitwise_and(ffrg,m[2])
  frg14 = cv2.bitwise_and(ffrg,m[3])
  frg15 = cv2.bitwise_and(ffrg,m[4])

  frg21 = cv2.bitwise_and(ffrg,m[5])
  frg22 = cv2.bitwise_and(ffrg,m[6])
  frg23 = cv2.bitwise_and(ffrg,m[7])
  frg24 = cv2.bitwise_and(ffrg,m[8])
  frg25 = cv2.bitwise_and(ffrg,m[9])

  frg31 = cv2.bitwise_and(ffrg,m[10])
  frg32 = cv2.bitwise_and(ffrg,m[11])
  frg33 = cv2.bitwise_and(ffrg,m[12])
  frg34 = cv2.bitwise_and(ffrg,m[13])
  frg35 = cv2.bitwise_and(ffrg,m[14])

  # Paint masks with each corresponding colour
  ffrgBGR = cv2.cvtColor(ffrg,cv2.COLOR_GRAY2BGR)
  ffrgBGRinv = cv2.bitwise_not(ffrgBGR)
  fr_bg = cv2.bitwise_and(fr,ffrgBGRinv)
  ffrgBGR_mcol = cv2.bitwise_and(mcol,ffrgBGR)
  fr_tot = cv2.add(fr_bg,ffrgBGR_mcol)
  
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
  rho11.append(a11/aVCA[0])
  rho12.append(a12/aVCA[1])
  rho13.append(a13/aVCA[2])
  rho14.append(a14/aVCA[3])
  rho15.append(a15/aVCA[4])

  rho21.append(a21/aVCA[5])
  rho22.append(a22/aVCA[6])
  rho23.append(a23/aVCA[7])
  rho24.append(a24/aVCA[8])
  rho25.append(a25/aVCA[9])

  rho31.append(a31/aVCA[10])
  rho32.append(a32/aVCA[11])
  rho33.append(a33/aVCA[12])
  rho34.append(a34/aVCA[13])
  rho35.append(a35/aVCA[14])

  # The traffic density on the total mask is the mean of the parts
  rho00.append(np.mean([
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
    
    del(rho00[0])
    
    nupdatei = nupdatei + 1
    
    if nupdatei == nupdate:
      nupdatei = 0
      
      # Calculate difference, propose by D.O.
#      drho21 = np.diff(rho21)
#      drho22 = np.diff(rho22)
      
      # Normaliza rho's
      rho21norm = np.asarray(rho21[-ncorr:])
      rho22norm = np.asarray(rho22[-ncorr:])
      rho21norm = rho21norm/np.sqrt(np.sum(rho21norm**2))
      rho22norm = rho22norm/np.sqrt(np.sum(rho22norm**2))
      c212 = np.correlate(rho21norm,rho22norm,'full')
      
      # Calculate cross correlation
      c112 = np.correlate(rho11[-ncorr:],rho12[-ncorr:],'full')
      c113 = np.correlate(rho11[-ncorr:],rho13[-ncorr:],'full')
      c114 = np.correlate(rho11[-ncorr:],rho14[-ncorr:],'full')
      c115 = np.correlate(rho11[-ncorr:],rho15[-ncorr:],'full')
      
      c121 = np.correlate(rho12[-ncorr:],rho11[-ncorr:],'full')
      c131 = np.correlate(rho13[-ncorr:],rho11[-ncorr:],'full')
      c141 = np.correlate(rho14[-ncorr:],rho11[-ncorr:],'full')
      c151 = np.correlate(rho15[-ncorr:],rho11[-ncorr:],'full')
      
#      c212 = np.correlate(rho21[-ncorr:],rho22[-ncorr:],'full')
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
      
      # Normalize correlation
      cnorm = [i+1 for i in range (ncorr-1)]+[ncorr-i for i in range (ncorr)]
      
      c112 = c112/cnorm
      c113 = c113/cnorm
      c114 = c114/cnorm
      c115 = c115/cnorm
      
      c212 = c212/cnorm
      c213 = c213/cnorm
      c214 = c214/cnorm
      c215 = c215/cnorm
      
      c312 = c312/cnorm
      c313 = c313/cnorm
      c314 = c314/cnorm
      c315 = c315/cnorm
      
      i = 2 #lane
      j = 2 #part
      if np.max(c212)>(np.mean(c212)+2*np.std(c212)):
        d212 = (np.argmax(c212)-ncorr-1)*fps
        s212.append(dMAP[i*4+j-1]/d212*3.6)
      else:
        d212 = 0
        s212.append(0)
        
      # Moving average propose by S.A.
#      if np.max(c212) > 0:
#        argavg_seba = np.average(range(ncorr*2-1),weights = c212)
#        d212 = (argavg_seba-(ncorr-1))*fps
#      else: 
#        d212 = 0
      
      # Calculate delay
      d112 = ((np.argmax(c112)-np.argmax(c121))/2.*fps if np.max(c112)>0 else 0)
      d113 = ((np.argmax(c113)-np.argmax(c131))/2.*fps if np.max(c112)>0 else 0)
      d114 = ((np.argmax(c114)-np.argmax(c141))/2.*fps if np.max(c112)>0 else 0)
      d115 = ((np.argmax(c115)-np.argmax(c151))/2.*fps if np.max(c112)>0 else 0)
      
#      d212 = ((np.argmax(c212)-np.argmax(c221))/2.*fps if np.max(c112)>0 else 0)
      d213 = ((np.argmax(c213)-np.argmax(c231))/2.*fps if np.max(c112)>0 else 0)
      d214 = ((np.argmax(c214)-np.argmax(c241))/2.*fps if np.max(c112)>0 else 0)
      d215 = ((np.argmax(c215)-np.argmax(c251))/2.*fps if np.max(c112)>0 else 0)
      
      d312 = ((np.argmax(c312)-np.argmax(c321))/2.*fps if np.max(c112)>0 else 0)
      d313 = ((np.argmax(c313)-np.argmax(c331))/2.*fps if np.max(c112)>0 else 0)
      d314 = ((np.argmax(c314)-np.argmax(c341))/2.*fps if np.max(c112)>0 else 0)
      d315 = ((np.argmax(c315)-np.argmax(c351))/2.*fps if np.max(c112)>0 else 0)
      
      # Average speed
      s112.append(dMAP[0]/d112*3.6) if d112 > 0 else s112.append(0)
      s113.append(dMAP[1]/d113*3.6) if d113 > 0 else s113.append(0)
      s114.append(dMAP[2]/d114*3.6) if d114 > 0 else s114.append(0)
      s115.append(dMAP[3]/d115*3.6) if d115 > 0 else s115.append(0)
      
#      s212.append(dMAP212/d212*3.6) if d212 > 0 else s212.append(0)
      s213.append(dMAP[5]/d213*3.6) if d213 > 0 else s213.append(0)
      s214.append(dMAP[6]/d214*3.6) if d214 > 0 else s214.append(0)
      s215.append(dMAP[7]/d215*3.6) if d215 > 0 else s215.append(0)
      
      s312.append(dMAP[8]/d312*3.6) if d312 > 0 else s312.append(0)
      s313.append(dMAP[9]/d313*3.6) if d313 > 0 else s313.append(0)
      s314.append(dMAP[10]/d314*3.6) if d314 > 0 else s314.append(0)
      s315.append(dMAP[11]/d315*3.6) if d315 > 0 else s315.append(0)
      
      # Plot
      ax0_lines[0][0].set_ydata(rho21)
      ax0_lines[1][0].set_ydata(rho22)
      
      ax1_lines[0].set_xdata(range(len(c212)))
      ax1_lines[0].set_ydata(c212)
      
      ax2_xdata = range(len(s112))
      ax2_lines[0].set_xdata(ax2_xdata)
      ax2_lines[0].set_ydata(s212)
      
      plt.draw()
      if len(s112) == nplot:
        del(s112[0])
        del(s113[0])
        del(s114[0])
        del(s115[0])
        
        del(s212[0])
        del(s213[0])
        del(s214[0])
        del(s215[0])
        
        del(s312[0])
        del(s313[0])
        del(s314[0])
        del(s315[0])

  # Info from capture
  pos = np.int32(cap.get(mod.CV_CAP_PROP_POS_FRAMES))

  # Show different steps
  cv2.putText(fr_tot,'%d/%d'%(pos,tot),(400,50),font,1,(0,0,255),2,cv2.LINE_AA)
  cv2.imshow('1 - Frame',fr_tot)
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

