# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:46:30 2015

@author: damzst
"""

# Librerías importadas
import cv2
import objetos_trayectorias as tray
import funciones_correccion as fun
import pickle

# Se cargan los resultados de loop_trayectorias.py
with open('trayectorias.pkl','rb') as input:
  trayectorias = pickle.load(input)

# Trayectoria a mostrar
indice = 0
if indice:
  trayectoriaElegida = trayectorias.trayectoriaPorIndice(indice)
else:
  trayectoriaElegida = trayectorias.mejorTrayectoria()
  indice = trayectoriaElegida.indice

# Video a analizar
cap = cv2.VideoCapture('viga.mp4')

# Inicio del video
cap.set(cv2.CAP_PROP_POS_FRAMES,
        10)
while cap.get < trayectoriaElegida.primerFotograma:
  ret,fotograma = cap.read()
  if not ret: break
    

# Cálculo de matrices globales del algoritmo de corrección
l,m = 1,952 # Parámetros del MUI
s = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) #Dimensión de la fuente
n = 1000 # Dimensión de esfera
w = 500 # Dimensión de salida
fov = 30 # FOV de salida
fun.calcularGlobales(l,m,s,n,w,fov)

# Lazo principal
for posicion in trayectoriaElegida.posiciones:
  # Se obtiene un fotograma
  ret,fotograma = cap.read()
  if not ret: break