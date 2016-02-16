# loop.py

# Librerias importadas
import cv2
import numpy as np

# Video a analizar
cap = cv2.VideoCapture('../../video/unq/agora18.mp4')

# Creacion del sustractor de fondo MOG2
bs = cv2.createBackgroundSubtractorMOG2()
bs.setDetectShadows(True)
bs.setShadowValue(0) #127
bs.setHistory(50) #500
#bs.setVarThreshold(36) #16
#bs.setBackgroundRatio(0.5) #0.9

# Tiempo en ms a esperar entre frames, 0 = para siempre
tms = 10

# Kernel
k_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
k_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))

# Filtro Kalman:
# Dimension de los estados: 4
# Dimension de la medicion: 2
# Dimension del vector de control: 0
#kalman = cv2.KalmanFilter(4,2,0)

# Loop principal
while(cap.isOpened()):
  # Obtener imagen
  frame = cv2.resize(cap.read()[1],(640,640))
  # Se aplica el sustractor de fondo
  msk = bs.apply(frame)
  # Se filtra el ruido
  blurmsk = cv2.GaussianBlur(msk,(21,21),0)
  frg = cv2.threshold(blurmsk,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
  # Funciones utiles para filtrar ruido
##  blurmsk = cv2.GaussianBlur(msk,(21,21),0)
##  frg = cv2.threshold(blurmsk,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
##  opening = cv2.morphologyEx(msk, cv2.MORPH_OPEN, k_opening)
##  closing = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, k_closing)
##  erode = cv2.erode(msk,k_erode,iterations = 1)
##  dilate = cv2.dilate(msk,k_dilate,iterations = 1)

  # Una vez obtenida una mascara decente, se obtienen los contornos de los blobs
  cnt = cv2.findContours(frg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
  #Se inicializan las listas M y centroides
  M,centroides = [],[]
  for i in range(len(cnt)):
    M.append(cv2.moments(cnt[i]))
    centroides.append(np.array([int(M[-1]['m10']/M[-1]['m00']),int(M[-1]['m01']/M[-1]['m00'])]) if M[-1]['m00']<>0 else np.zeros(2,int))

  cv2.drawContours(frame,cnt,-1,(0,0,255),2)
  # Se muestran los distintos pasos del algoritmo
  cv2.imshow('Frame',frame)
#  cv2.imshow('Blur',blur)
  # Se mueve la ventana
  cv2.moveWindow('Frame',0,0)
#  cv2.moveWindow('Blur',480,0)
  # Botones de terminado, pausado, reanudar
  k = cv2.waitKey(tms) & 0xFF
  if k == ord('q'):
    break # Terminar
  elif k == ord('p'):
    tms = 0 #Pausar
  elif k == ord('f'):
    tms = 10 # Reanudar

# Liberar el video y destruir las ventanas
cap.release()
cv2.destroyAllWindows()
