# Our code, translated from Matlab to python, albeit slightly changed

# Imported libraries
import numpy as np
import cv2

# Folder and videos to choose
mainFolder = '/home/damzst/Documents/tpf/video'
vidBalkonWinter = '/balkon/balkonWinter_red760.avi'
vidBalkonSummer = '/balkon/balkonSummer.mp4'
vidBalkonSummerRed = '/balkon/balkonSummer_red.avi'
chosenOne = vidBalkonSummerRed
cap = cv2.VideoCapture(mainFolder+chosenOne)

# Sometimes there are useless frames at the beginning
uselessframes = 200
cap.set(1,uselessframes)

# Optimal parameters
fps = 10.
tau = 10.
alpha = 1/(fps*tau)
#alpha = 0.0001
U1 = 50
U2 = 50
sqNvec = 6

# Kernel definition
kernel = np.ones((sqNvec,sqNvec),np.float32)/sqNvec**2

# First frames
mu = cap.read()[1]

# Main loop
while(cap.isOpened()):
  # Read next frame
  img = cap.read()[1]
  # Differences between frames
  dif = cv2.absdiff(img,mu)
  # Convert to gray
  difgray = cv2.cvtColor(dif,cv2.COLOR_BGR2GRAY)
  # Use U1 as threshold
  hij = cv2.threshold(difgray,U1,255,cv2.THRESH_BINARY)[1]
  # Filter result
  win = cv2.filter2D(np.float32(hij),-1,kernel)
  # Use U2 as threshold
  cij = np.uint8(cv2.threshold(win,U2,255,cv2.THRESH_BINARY)[1])
  cijbgr = cv2.cvtColor(cij,cv2.COLOR_GRAY2BGR)
  # Update mu - Accumulate Weighted
  mu = np.uint8(alpha*img+(1-alpha)*mu)
  # Show images
  cv2.imshow('0 - img',img)
  cv2.imshow('1 - dif',dif)
  cv2.imshow('2 - hij',hij)
  cv2.imshow('3 - win',cv2.convertScaleAbs(win))
  cv2.imshow('4 - cij',cv2.bitwise_and(img,cijbgr))
  cv2.imshow('mu',mu)
  # You need to keep pressing a key to process,
  # and press 'q' to stop.
  if cv2.waitKey(10) & 0xFF == ord('q'):
    break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()



