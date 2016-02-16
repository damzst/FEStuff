# Accumulate Weighted to get BKG

# Imported libraries
import numpy as np
import cv2

# Folder and videos to choose
mainFolder = '/home/damzst/Documents/tpf/video'
vidBalkonSummerRed = '/balkon/balkonSummer_red.avi'

# Open video file
chosenOne = vidBalkonSummerRed
cap = cv2.VideoCapture(mainFolder+chosenOne)

# First frame
ret,frame=cap.read()

# Accumulator
acc = np.zeros(frame.shape,np.float32)
tot = np.float32(cap.get(7))
pos = np.int32(cap.get(1))

# Main Loop
while(ret):
  pos = np.int32(cap.get(1))
  # Accumulate
  acc = acc+np.float32(frame)
  
  # Read next frame
  ret,frame = cap.read()
  print cap.get(1),np.max(acc/tot)

  # Calculate division
  #acc = acc/tot
  aux = acc/tot

  # Show mean
  mu1 = cv2.convertScaleAbs(aux)
#  mu2 = np.uint8(acc)

#  print np.max(mu1),mu1.dtype,np.max(mu2),mu2.dtype

  cv2.imshow('mu1',mu1)
#  cv2.imshow('mu2',mu2)
  
  if cv2.waitKey(5) & 0xFF == ord('q'):
    break


## Calculate division
##acc = acc/tot
#acc = acc/5000.

## Show mean
#mu1 = cv2.convertScaleAbs(acc)
#mu2 = np.uint8(acc)

#print np.max(mu1),mu1.dtype,np.max(mu2),mu2.dtype

#cv2.imshow('mu1',mu1)
#cv2.imshow('mu2',mu2)

# You need to keep pressing a key to process,
# and press 'q' to stop.
#while(1):
#  if cv2.waitKey(0) & 0xFF == ord('q'):
#    break

# Save it
#cv2.imwrite('mu1.png',mu1)
#cv2.imwrite('mu2.png',mu2)

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
