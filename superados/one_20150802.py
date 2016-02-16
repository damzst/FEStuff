# main.py

# Librerias importadas
import fun
import cv2

# Para medir tiempo
import time
#start = time.clock()
# En esta posicion iria el codigo a evaluar
#print start-time.clock()

# Se carga la imagen tomada con la camara VIVOTEK FE8172
src = cv2.imread('./resources/IM1.jpg')

# Parametros del Modelo Unificado de Imagenes
l,m =  1,952
u0,v0 = 960,960
# Dimension de la proyeccion esferica deseada (nxn)
n = 1000
# Se mapean los puntos desde la esfera unitaria hacia el plano imagen VCA
U,V = fun.map_sph(n,l,m,u0,v0)
# Se proyectan los puntos desde el plano imagen VCA hacia la esfera unitaria
start = time.clock()
sph = cv2.remap(src,U,V,cv2.INTER_NEAREST)
print time.clock()-start

# Punto de interes:
u,v = 350,1100
# Se calcula la matriz de rotacion para apuntar la esfera a un punto (u,v) dado
MR = fun.findMR(u,v,l,m,u0,v0)
# Se mapean los puntos desde la esfera rotada hacia la esfera original
Theta,Phi = fun.map_sphrot(1000,MR)
# Se proyectan los puntos desde la esfera original hacia la rotada
start = time.clock()
sphrot = cv2.remap(sph,Theta,Phi,cv2.INTER_NEAREST)
print time.clock()-start

# Campo de vision deseado en [deg]
fov = 30
# Dimension de la imagen en perspectiva deseada
w = 500
# Se mapean los puntos desde el plano imagen PTZ virtual hacia la esfera rotada
ThetaR,PhiR = fun.map_ptzv(w,n,fov)
# Se proyectan los puntos desde la esfera rotada hacia el plano imagen PTZ virtual
start = time.clock()
ptzv = cv2.remap(sphrot,ThetaR,PhiR,cv2.INTER_LINEAR)
print time.clock()-start

# Se muestra la imagen PTZ virtual obtenida
#cv2.imshow('Imagen original',cv2.resize(src,(500,500)))
#cv2.imshow('Imagen esferica',cv2.resize(sph,(500,500)))
cv2.imshow('Imagen esferica rotada',cv2.resize(sphrot,(500,500)))
cv2.imshow('Imagen PTZ virtual obtenida',ptzv)
# Se reubican las ventanas
cv2.moveWindow('Imagen esferica rotada',548,0)
cv2.moveWindow('Imagen PTZ virtual obtenida',0,0)
# Limpieza de las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()
