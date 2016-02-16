# loop.py

# Librerias importadas
import cv2
import numpy as np
from munkres import Munkres
from matplotlib import pyplot as plt

class Trayectoria(object):
  """
  Conjunto de atributos que definen la trayectoria
  Atributos:
    idx: indice que lo identifica, debe ser unico
    meas: lista de posiciones medidas asignadas
    pred: lista de posiciones predichas por el filtro
    kf: filtro de Kalman asociado
    li: ultima iteracion en la que se asigno una nueva posicion
  """
  def __init__(self,pos,iteracion,idx):
    # Asigno indice
    self.idx = idx
    # Inicializo listas
    self.meas = []
    self.pred = []
    # Filtro Kalman:
    # Dimension de los estados: 4
    # Dimension de la medicion: 2
    # Dimension del vector de control: 0
    self.kf = cv2.KalmanFilter(4,2,0)
    # Matrices del filro
    self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    self.kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    self.kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
    # Asigno la posicion inicial
    tmpState = np.float32(self.kf.statePre)
    tmpState[0] = pos[0]
    tmpState[1] = pos[1]
    self.kf.statePre = np.float32(tmpState)
    self.new_pos(pos,iteracion)

  def new_pos(self,pos,iteracion):
    # Asigno 
    self.kf.correct(np.array([[np.float32(pos[0])],[np.float32(pos[1])]]))
    self.li = iteracion
    # Agrego a la lista
    self.meas.append(pos)
    self.pred.append(self.kf.predict())

# Creacion del objeto Munkres que asigna los nuevos puntos a las trayectorias
m = Munkres()

# Video a analizar
cap = cv2.VideoCapture('../../video/unq/viga.mp4')

# Creacion del sustractor de fondo MOG2
bs = cv2.createBackgroundSubtractorMOG2()
bs.setDetectShadows(True)
bs.setShadowValue(0) #127
#bs.setHistory(50) #500
#bs.setVarThreshold(36) #16
#bs.setBackgroundRatio(0.5) #0.9

# Tiempo en ms a esperar entre frames, 0 = para siempre
tms = 10

# Kernel
##k_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
##k_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))

# Flag numero de iteracion
iteracion = 0

# Primer iteracion
primer = 0

# Lista de trayectorias (objetos)
trayectorias = []

# Loop principal
while(cap.isOpened()):
  # Obtener imagen
  frame = cv2.resize(cap.read()[1],(640,640))
#  frame = cap.read()[1]
  # Se aplica el sustractor de fondo
  msk = bs.apply(frame)
  # Se filtra el ruido
#  blurmsk = cv2.GaussianBlur(msk,(21,21),0)
#  frg = cv2.threshold(blurmsk,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
  blurmsk = cv2.GaussianBlur(msk,(17,17),0)
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
  # Se inicializan las listas M y centroides
  M,meas,areas = [],[],[]
#  M,meas = [],[]
  for i in range(len(cnt)):
    M.append(cv2.moments(cnt[i]))
    meas.append(np.array([int(M[-1]['m10']/M[-1]['m00']),int(M[-1]['m01']/M[-1]['m00'])]) if M[-1]['m00']<>0 else np.zeros(2,int))
    areas.append(M[-1]['m00'])
  
  cv2.drawContours(frame,cnt,-1,(0,0,255),2)
  
# Quiero pintar los 20 blobs mas grandes
#  areas = np.array(areas)
#  idx = np.argsort(areas)
#  
  
  
#  areas,meas,cnt = [list(x) for x in zip(*sorted(zip(areas,meas,cnt))]
#  areas,meas,cnt = areas[:N],meas[:N],cnt[:N]

#  cv2.drawContours(frame,[x[2] for x in amc],-1,(0,255,0),2)

#  print 'promedio',np.mean(areas)
#  print 'iteracion',iteracion,'meas',len(meas)
  
  print len(areas)
  if len(meas) > 0:
    if primer == 0:
      # En la primer iteracion, todos los objetos detectados generan un objeto trayectoria
      for i in range(len(meas)):
        trayectorias.append(Trayectoria(meas[i],iteracion,i))
        primer = 1
    else:
      # A partir de la segunda iteracion, se obtiene la matriz de costos entre las posiciones predichas de las trayectorias y las nuevas posiciones detectadas
      pred = [] # Lista de predicciones
      costs = [] # Lista de listas de costos (al final es una matriz)
      for i in range(len(trayectorias)):
        # De cada una de las trayectorias se toma la prediccion, se agrega a "pred"
        pred.append(trayectorias[i].pred[-1])
        # Se inicializa la lista de costos (distancias) para cada una de las combinaciones entre la prediccion y todos los puntos detectados "meas"
        costsi = []
        for j in range(len(meas)):
          # Se calcula la distancia
          costsi.append(np.sqrt((pred[-1][0]-meas[j][0])**2+(pred[-1][1]-meas[j][1])**2))
        # Se agrega la lista de costos asociados a esta prediccion a la lista general
        costs.append(costsi)
      print 'cost',len(costs)
      # Una vez terminada la matriz de costos, se usa el algoritmo de Munkres para asignar los puntos
      indexes = m.compute(costs)
      for row,column in indexes:
        # Se busca el valor del costo de asignar la fila (pred) con la columna (meas)
        value = costs[row][column]
        if value>20:
          # Si el costo es mayor a 20 (distancia mayor a 20 pixeles), se crea un objeto nuevo
          trayectorias[row].new_pos(meas[column],iteracion)
        else:
          # Si el costo es menor, se considera que es correcto asignar el punto detectado a la trayectoria correspondiente a la prediccion
          trayectorias.append(Trayectoria(meas[column],iteracion,trayectorias[-1].idx))
      for i in range(len(trayectorias)):
        # Se eliminan las trayectorias viejas
        age = iteracion-trayectorias[i].li
        if age>10:
          # Si pasan mas de 10 iteraciones sin asignar ningun punto nuevo, se elimina la trayectoria
          del(trayectorias[i])
  
  iteracion = iteracion +1
  
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

  # Ahora quiero graficar las trayectorias:
  # Empiezo graficando una:

#  i = 0
#  traypred = np.asarray(trayectorias[i].pred)
#  traymeas = np.asarray(trayectorias[i].meas)
#  
#  plt.plot(traypred[:,0],traypred[:,1],'xr',label='Predicho')
#  plt.axis([0,1920,0,1920])
#  plt.hold(True)
#  plt.plot(traymeas[:,0],traymeas[:,1],'ob',label='Mediciones')
#  plt.legend(loc=2)
#  plt.title("Filtro Kalman Velocidad Cte")
#  plt.show()

# Liberar el video y destruir las ventanas
cap.release()
cv2.destroyAllWindows()
