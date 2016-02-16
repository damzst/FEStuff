# Find which set of parameters throws less error.
# Labelling all is too time-consuming to apply.
# The winner will be the one with the following properties:
# Mean of objects by frame nearer 10 (arbitrary) 

# Imported libraries
import numpy as np
import cv2

# Folder and videos to choose
mainFolder = '/home/damzst/Documents/tpf/video'
vidBalkonWinter = '/balkon/balkonWinter_red760.avi'

# Parameters
alphav = np.linspace(0.1,0.5,5)
U1v = np.linspace(10,40,4)
U2v = np.linspace(10,40,4)
sqNvecv = np.linspace(10,60,6)

# Results
results = open('Results.txt','w')
results.write('alpha U1 U2 sqNvec\n') 	

# Loops for each combination
for alpha in alphav:
	for U1 in U1v:
		for U2 in U2v:
			for sqNvec in sqNvecv:
				# Capture the chosen video
				cap = cv2.VideoCapture(mainFolder+vidBalkonWinter)
				# Kernel definition
				kernel = np.ones((sqNvec,sqNvec),np.float32)/sqNvec**2
				# First frames
				mu = cap.read()[1]
				# Frame count
				frameCount = 0
				# Blob count
				blobCount = 0.
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
					win = cv2.filter2D(hij,-1,kernel)
					# Use U2 as threshold	
					cij = cv2.threshold(win,U2,255,cv2.THRESH_BINARY)[1]
					# Find blobs
					aux = cij.copy()
					contours = cv2.findContours(aux,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
					blobCount = blobCount+len(contours)
					# Update mu	
					mu = np.uint8(alpha*img+(1-alpha)*mu)	
					# Show images	
					cv2.imshow('cij',cij)
					# Press 'q' to stop. 
					if cv2.waitKey(20) & 0xFF == ord('q'):
						break
					# Just process some frames
					frameCount = frameCount+1
					if frameCount==200:
						break
				# Release everything if job is finished
				cap.release()
				cv2.destroyAllWindows()
				# Write Results
				results.write(str(alpha)+' '+str(U1)+' '+str(U2)+' '+str(sqNvec)+' '+str(blobCount)+'\n')
# Close text file
results.close()



