# -*- coding: utf-8 -*-
"""
dummy.py

@author: Administrator
"""

# Librerías importadas
import cv2
import objetos_trayectorias as tray

# Video a analizar
cap = cv2.VideoCapture('viga.mp4')
# Mascara utilizada
mascara = cv2.imread('viga_mascara.png')
# Creación del sustractor de fondo MOG2
bs = cv2.createBackgroundSubtractorMOG2()
bs.setDetectShadows(True)
bs.setShadowValue(0)

# Tiempo en ms a esperar entre frames, 0 = para siempre
tms = 0
# Inicio del video
cap.set(1,10)

# Lazo principal
while(cap.isOpened()):
  # Se obtiene un fotograma
  numeroDeFotograma = cap.get(1)
  ret,fotograma = cap.read()
  if not ret: break
  fotogramaEscalado = cv2.resize(fotograma,(640,640))
  fotogramaConMascara = cv2.bitwise_and(fotogramaEscalado,mascara)
  # Se aplica el sustractor de fondo
  frgSinFiltrar = bs.apply(fotogramaConMascara)
  # Se filtra el ruido
  frgConBlur = cv2.GaussianBlur(frgSinFiltrar,(17,17),0)
  frg = cv2.threshold(frgConBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
  # Se obtienen los contornos de los blobs
  contornos = cv2.findContours(frg.copy(),
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)[1]
  frgblob = fotogramaEscalado.copy()
  cv2.drawContours(frgblob,
                   contornos,
                   -1,
                   (0,0,255),
                   2)
  

  # Se muestran los blobs y los índices
  cv2.imshow('Fotograma',fotogramaEscalado)
  # Botones de terminado, pausado, reanudar
  k = cv2.waitKey(tms) & 0xFF
  if k == ord('q'):
    break # Terminar
  elif k == ord('p'):
    tms = 0 #Pausar
  elif k == ord('f'):
    tms = 10 # Reanudar

# Se guardan los resultados
cv2.imwrite('det_framesc.jpg',fotogramaEscalado)
cv2.imwrite('det_frgmog2.jpg',frgSinFiltrar)
cv2.imwrite('det_frgblur.jpg',frgConBlur)
cv2.imwrite('det_frgotsu.jpg',frg)
cv2.imwrite('det_frgblob.jpg',frgblob)

# Liberar el video y destruir las ventanas
cap.release()
cv2.destroyAllWindows()
