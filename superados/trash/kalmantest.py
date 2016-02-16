import numpy as np
import cv2
import random
from matplotlib import pyplot as plt

kalman = cv2.KalmanFilter(4,2,0)

# My trayectory
N = 100
sigma = .1
points = np.zeros((N,2),np.float32)
real = np.zeros((N,2),np.float32)
for n in range(N):
  real[n,0] = 1*n
  real[n,1] = 0*n
  points[n,0] = real[n,0] + random.gauss(0,sigma)
  points[n,1] = real[n,1] + random.gauss(0,sigma)

# Initial conditions
xinit = points[0,0]
yinit = points[0,1]
vxinit = 0
vyinit = 0
kalman.statePre = np.array([[xinit],[yinit],[vxinit],[vyinit]],np.float32)

# Kalman parameters
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.processNoiseCov = np.float32(.01*np.eye(4)) #Q
kalman.measurementNoiseCov = np.float32(1*np.eye(2)) #R
kalman.errorCovPost = np.float32(1*np.ones((4,4))) #P(k-1) initial a posteriori

# Predictions and corrections
predictions = np.zeros((N-1,4),np.float32)
corrections = np.zeros((N-1,4),np.float32)
for n in np.arange(1,N):
  print np.transpose(kalman.statePre)
  predictions[n-1] = np.transpose(kalman.predict())
  corrections[n-1] = np.transpose(kalman.correct(np.transpose(np.float32(points[n]))))

# Plot
plt.plot(predictions[:,0],predictions[:,1],'-xr',label='Predicted')
#plt.axis([-30,-30,-30,-30])
#    plt.hold(True)
plt.plot(corrections[:,0],corrections[:,1],'-ob',label='Corrected')
plt.plot(points[:,0],points[:,1],'-xg',label='Measured')
plt.plot(real[:,0],real[:,1],'--xb',label='Real')
plt.legend(loc=2)
plt.title("Kalman Filter Results")
plt.show()





