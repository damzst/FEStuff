# FindContour

# Imported libraries
import numpy as np
import cv2

# Load screenshot
scr = cv2.imread('scr.png',0)

# Contours
_,contours,hierarchy = cv2.findContours(scr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

epsilon = []
approx = []

for cnt in contours:
  epsilon.append(0.1*cv2.arcLength(cnt,True))
  approx.append(cv2.approxPolyDP(cnt,epsilon[-1],True))


#M = cv2.moments(cnt)
#for x in M.items():
#  print x

#cx = int(M['m10']/M['m00'])
#cy = int(M['m01']/M['m00'])


print contours
# Draw contours
cv2.drawContours(scr,approx,-1,(255,255,255),3)
cv2.imshow('scr',scr)
while(1):
  if cv2.waitKey(10) & 0xFF == ord('q'):
    break
cv2.destroyAllWindows()




