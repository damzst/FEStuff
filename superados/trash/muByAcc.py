# Accumulate Weighted to get BKG

# Imported libraries
import numpy as np
import cv2

# Folder and videos to choose
mainFolder = '/home/damzst/Documents/tpf/video'
vidBalkonWinter = '/balkon/balkonWinter_red760.avi'
vidBalkonSummerRed = '/balkon/balkonSummer_red.avi'

# Open video file
chosenOne = vidBalkonSummerRed
cap = cv2.VideoCapture(mainFolder+chosenOne)

# First frame
frame = cap.read()[1]

# Accumulator
acc = np.float32(frame)

# Main Loop
while(cap.isOpened()):
  # Read next frame
  frame = cap.read()[1]
  # Accumulate
  cv2.accumulateWeighted(frame,acc,0.01)
  # Rescale
  mu = cv2.convertScaleAbs(acc)
  # Show mean	
  cv2.imshow('mu',mu)
  pos = np.int32(cap.get(1))
  tot = np.int32(cap.get(7))
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(frg,str(pos),(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)
  cv2.putText(bkg,str(pos),(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)
  cv2.putText(frg,str(tot),(10,25), font, 1,(255,255,255),2,cv2.LINE_AA)
  cv2.putText(bkg,str(tot),(10,25), font, 1,(255,255,255),2,cv2.LINE_AA)
  # You need to keep pressing a key to process,
  # and press 'q' to stop. 
  if cv2.waitKey(0) & 0xFF == ord('q'):
    break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
