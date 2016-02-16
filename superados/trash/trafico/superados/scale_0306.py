# Scales the video to a fourth of the original size.
# Given that VIVOTEK FE captures at 1920x1920, we'll get 480x480.
# This will reduce the processing time on tests.

# Imported libraries
import numpy as np
import cv2

# Folder and videos to choose
mainFolder = '/home/damzst/Documents/tpf/video'
vidUNQHall = '/unq/UNQHall'
vidUNQBridge = '/unq/UNQBridge'
vidBalkonSummer = '/balkon/balkonSummer'
vidBalkonWinter = '/balkon/balkonWinter'

# Open video chosen file
chosenOne = vidBalkonSummer
cap = cv2.VideoCapture(mainFolder+chosenOne+'.mp4')
# Sometimes there are useless frames at the beginning
uselessframes = 80
for _ in range(uselessframes):
  _,_ = cap.read()

# Define the desired resolution, codec and create VideoWriter object
res = 760
scale = np.float32(res/cap.get(3))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(mainFolder+chosenOne+'_red.avi',fourcc, 20.0, (res,res))

# Main Loop
while(cap.isOpened()):
  # Read frame
  ret, frame = cap.read()
  if ret==True:
    # Scale down video to a fourth 
    red = cv2.resize(frame,None,fx=scale,fy=scale,
    interpolation=cv2.INTER_CUBIC)
    # Show the result
    cv2.imshow('Reduced Image',red)
    # Save it
    out.write(red)
    # Press 'q' to stop. 
    if cv2.waitKey(20) & 0xFF == ord('q'):
      break
  else:
    break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

