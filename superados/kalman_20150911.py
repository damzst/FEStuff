# kalman.py

# Librerias importadas
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

# Se genera una serie de puntos
N = 50 #Muestras
x = np.linspace(0.,1000.,N) #(Comienzo,Fin,Muestras)
y = np.linspace(0.,1000.,N) #(Comienzo,Fin,Muestras)

# Se le suma el ruido
xns = 10#Magnitud del ruido en x
yns = 10#Magnitud del ruido en y
xn = x+(np.random.rand(N)-0.5)*2*xns
yn = y+(np.random.rand(N)-0.5)*2*yns

# Se crea el filtro Kalman
kf = cv2.KalmanFilter(4,2,0)

# Se definen las matrices del filtro
kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03

# Se inicializa el filtro Kalman
kf.statePre[0,0] = xn[1]
kf.statePre[1,0] = yn[1]
kf.statePre[2,0] = xn[1]-xn[0]
kf.statePre[3,0] = yn[1]-yn[0]

# Lista para guardar las mediciones y las predicciones
meas = []
pred = []

# Loop
for i in range(N-2):
  mp = np.array([[np.float32(xn[i+2])],[np.float32(yn[i+2])]])
  kf.correct(mp)
  tp = kf.predict()
  meas.append(mp.transpose())
  pred.append(tp.transpose())

measn = np.squeeze(np.asarray(meas))
predn = np.squeeze(np.asarray(pred))

plt.plot(measn[:,0],measn[:,1],'xr',label='Medido')
plt.axis([0,1100,0,1100])
plt.hold(True)
plt.plot(predn[:,0],predn[:,1],'ob',label='Salida del Filtro')
plt.legend(loc=2)
plt.title("Filtro Kalman Velocidad Cte")
plt.show()


