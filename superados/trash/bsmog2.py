# Script to compare our simple algorithm to those in the library

# Imported libraries
import numpy as np
import cv2

# Capture the chosen video
mainFolder = '/home/damzst/Documents/tpf/video'
vidBalkonWinter = '/balkon/balkonWinter_red760.avi'
vidBalkonSummer = '/balkon/balkonSummer.mp4'
vidBalkonSummerRed = '/balkon/balkonSummer_red.avi'
chosenOne = vidBalkonSummerRed
cap = cv2.VideoCapture(mainFolder+chosenOne)

# Sometimes there are useless frames at the beginning
uselessframes = 200
cap.set(1,uselessframes)

# Load mask
mask = cv2.imread(mainFolder+chosenOne[:-4]+'_mask.png')



# Gaussian blur kernel size
ks = 7

# I'll save the bkg and frg obtained by bs
#res = np.int32(cap.get(3))
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#frgout = cv2.VideoWriter(mainFolder+chosenOne[:-4]+'_frg.avi',fourcc, 20.0, (res,res))
#bkgout = cv2.VideoWriter(mainFolder+chosenOne[:-4]+'_bkg.avi',fourcc, 20.0, (res,res))

# Results
#results = open('MOG2InTime.txt','w')
#results.write('MOG2 parameter evolution\nHist NMix BKGR VT VTG VI CRT ST\n')

# MOG2 parameters
#cuenta = 1
#bs.setHistory(cuenta) #T->alpha=1/T
bs.setHistory(7500)
#bs.setBackgroundRatio(0.7) #1-cf
#bs.setComplexityReductionThreshold(0)
bs.setVarThreshold(36)
#bs.setBackgroundImage(cv2.imread('mu.png'))


#bs2.setHistory(7500) #T->alpha=1/T
#bs2.setVarThreshold(36)

#0 CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds or video capture timestamp.
#1 CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
#2 CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file: 0 - start of the film, 1 - end of the film.
#3 CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
#4 CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
#5 CV_CAP_PROP_FPS Frame rate.
#6 CV_CAP_PROP_FOURCC 4-character code of codec.
#7 CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
#8 CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
#9 CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
#10 CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
#11 CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
#12 CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
#13 CV_CAP_PROP_HUE Hue of the image (only for cameras).
#14 CV_CAP_PROP_GAIN Gain of the image (only for cameras).
#15 CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
#16 CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
#17 CV_CAP_PROP_WHITE_BALANCE Currently not supported
#18 CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)


#frg = np.zeros((cap.get(4),cap.get(3)))
  #####################################################
  # MOG2 parameters
#  phist = bs.getHistory() #500 T
#  pnmix = bs.getNMixtures() #5 M
#  pbkgr = bs.getBackgroundRatio() #0.9 cthr
#  pvt = bs.getVarThreshold() #16
#  pvtg = bs.getVarThresholdGen() #9
#  pvi = bs.getVarInit() #15 
#  pcrt = bs.getComplexityReductionThreshold() #0.05 ct
#  pst = bs.getShadowThreshold() #0.5
  
  # Write Results
#  results.write(str(phist)+' '+str(pnmix)+' '+str(pbkgr)+' '+str(pvt)+' '+str(pvtg)+' '+str(pvi)+' '+str(pcrt)+' '+str(pst)+'\n')
  #####################################################

#ko = 2
#ka = 5
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ko,ko))

#kernelo = np.ones((ko,ko),np.uint8)
#kernela = np.ones((ka,ka),np.uint8)

# Create the BKG substractors from the library
bs = cv2.createBackgroundSubtractorMOG2()
bs.setDetectShadows(True)
bs.setShadowValue(10)

#bs2 = cv2.createBackgroundSubtractorMOG2()
#bs2.setDetectShadows(True)
#bs2.setShadowValue(0)
# Main Loop
while(cap.isOpened()):
  # Read frame  
  frame = cap.read()[1]
  # Apply mask
  fr2 = cv2.bitwise_and(frame,mask)
  # Apply BKG substractor
  frg = bs.apply(fr2)
  # Get Background
  bkg = bs.getBackgroundImage()

#  frg = cv2.erode(frg,kernel,iterations = 1)
  # Sth I tried to smooth the foreground
  ret = cv2.erode(frg,kernelo,iterations = 1)
  ret = cv2.dilate(ret,kernela,iterations = 1)
#  ret2 = cv2.erode(frg2,kernel,iterations = 1)
#  ret = cv2.dilate(erode,kernel,iterations = 1)
#  ret = cv2.morphologyEx(frg2, cv2.MORPH_OPEN, kernel)
#  blur = cv2.GaussianBlur(opening,(ks,ks),0)
#  ret = cv2.threshold(blur,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

  # Add frame count
  pos = np.int32(cap.get(1))
  tot = np.int32(cap.get(7))
  mseg = np.float32(cap.get(0))
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(frg,str(pos),(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)
  cv2.putText(bkg,str(pos),(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)
  cv2.putText(frg,str(tot),(10,25), font, 1,(255,255,255),2,cv2.LINE_AA)
  cv2.putText(bkg,str(tot),(10,25), font, 1,(255,255,255),2,cv2.LINE_AA)
  cv2.putText(bkg,str(mseg/1000),(10,75), font, 1,(255,255,255),2,cv2.LINE_AA)
  # Show different steps
#  cv2.imshow('1 - Frame',fr2)
  cv2.imshow('2 - Foreground',frg)
  cv2.imshow('2 - Foreground2',frg2)
#  cv2.imshow('3 - Smoothed   Frg',ret)
#  cv2.imshow('4 - Background',bkg)

  # Save videos
#  frgBGR = cv2.cvtColor(frg,cv2.COLOR_GRAY2BGR)
#  frgout.write(frgBGR)
#  bkgout.write(bkg)

  # Keep pressing any key to continue processing
  # Press 'q' to stop
  if cv2.waitKey(10) & 0xFF == ord('q'):
    break

# Release everything
#results.close()
cap.release()
#frgout.release()
#bkgout.release()
cv2.destroyAllWindows()


