# Some useful functions for our VCA image processing

import numpy as np
from numpy import matrix

# Forgotten OpenCV constants:

CV_CAP_PROP_POS_MSEC = 0
CV_CAP_PROP_POS_FRAMES = 1
CV_CAP_PROP_POS_AVI_RATIO = 2
CV_CAP_PROP_FRAME_WIDTH = 3
CV_CAP_PROP_FRAME_HEIGHT = 4
CV_CAP_PROP_FPS = 5
CV_CAP_PROP_FOURCC = 6
CV_CAP_PROP_FRAME_COUNT = 7
CV_CAP_PROP_FORMAT = 8
CV_CAP_PROP_MODE = 9
CV_CAP_PROP_BRIGHTNESS = 10
CV_CAP_PROP_CONTRAST = 11
CV_CAP_PROP_SATURATION = 12
CV_CAP_PROP_HUE = 13
CV_CAP_PROP_GAIN = 14
CV_CAP_PROP_EXPOSURE = 15
CV_CAP_PROP_CONVERT_RGB = 16
CV_CAP_PROP_WHITE_BALANCE_U = 17
CV_CAP_PROP_WHITE_BALANCE_V = 18
CV_CAP_PROP_RECTIFICATION = 19
CV_CAP_PROP_ISO_SPEED = 20
CV_CAP_PROP_BUFFERSIZE = 21

def vca2map(xi,yi):
  "Transformation from FE image to MAP image."
  # Cuaternion transformation parameters
  # Displacement
  Px,Py,Pz = 297.404047304,272.780351563,29.7903688186
  # Rotation matrix, already calculated for desired angles
  T11,T12,T13 = -0.11797354453155016, 0.99235949420581748, -0.030520269689462087
  T21,T22,T23 = 0.99273277781555358, 0.11873163593465613, 0.01960689875770763
  T31,T32,T33 = 0.023746044750169993, -0.033547472152973622, -0.99915498921381984
  
  # Auxiliary calculations (only useful for 760x760px images)
  xcen, ycen = xi-380, yi-380
  r = np.sqrt(xcen**2+ycen**2)
  phi = np.arctan2(ycen,xcen)
  
  # Transform to spherical coordinates
  theta = 2*np.arctan(r*0.002653247132282058)
  
  # Calculate sines and cosines once
  ct = np.cos(theta)
  st = np.sin(theta)
  cp = np.cos(phi)
  sp = np.sin(phi)
  
  # Now the coordinates on the map
  rho = -Pz/(st*(T31*cp+T32*sp)+ct*T33)
  xm = rho*(st*(T11*cp+T12*sp)+T13*ct)+Px
  ym = rho*(st*(T21*cp+T22*sp)+T23*ct)+Py

  return xm,ym


def map2vca(xm,ym):
  "Transformation from MAP image to FE image."
  # Displacement, negative respect to vca2map
  Px,Py,Pz = -297.404047304,-272.780351563,-29.7903688186
  # Rotation Matrix
  T_1 = np.array([[-0.11782417,  0.99275255,  0.02367077],
       [ 0.99256275,  0.11847825, -0.02853952],
       [-0.03687426,  0.02013208, -0.9993288 ]])
  
  # Displace to camera origin
  Xaux = np.matrix([[xm+Px],[ym+Py],[Pz]])
  # Rotate to get in camera reference frame
  XC = T_1.dot(Xaux) # Now in camera cartesian coordinates
  
  fi = np.arctan2(XC[1],XC[0]) # Angle in image
  t = np.arctan2(np.sqrt(XC[0]**2+XC[1]**2),XC[2]) 
  r = np.tan(t/2)*376.8966666666666 # Radius in image
  
  # Call 'np.float' to avoid 'np.matrix' type, in output
  xi = np.float(380 + r*np.cos(fi))
  yi = np.float(380 + r*np.sin(fi))
  
  return xi,yi

def areaMap(m,px_m = 0.8):
  '''
  Calculate the area in MAP image of the entry FE mask (m).
  It uses the argument px_m = px/m to this end.
  '''
  aux = m[:,:,1].copy()
  c = cv2.findContours(aux,cv2.RETR_EXTERNAL,2)[1]
  # Load list as array and remove 1-dim entries
  ca = np.squeeze(c)
  pMap = []
  for i in range(len(ca)):
    xm, ym = np.int32(vca2map(ca[i,0],ca[i,1]))
    pMap.append((xm,ym))
  areapx2 = cv2.contourArea(np.asarray(pMap))
  aream2 = areapx2 / px_m**2
  return aream2


