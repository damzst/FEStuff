# -*- coding: utf-8 -*-
"""
Aca se definen funciones basicas para hacer la calibracion y transformaciones al pasar de una imagen VCA a un mapa.

@author : sarroyo
"""

import numpy as np
from lmfit import Parameters
import pickle

# se define la transformacion desde VCA a mapa, devuelve las dos coordenadas
def ForwardVCA2Map(parametros,xI,yI):
	"""
	Descripcion de la funcion.
	"""
	cen=parametros['cen'].value
	k=parametros['k'].value
	alfa=parametros['alfa'].value
	beta=parametros['beta'].value
	gamma=parametros['gamma'].value
	Px=parametros['Px'].value
	Py=parametros['Py'].value
	Pz=parametros['Pz'].value
	
	##calculos auxiliares sobre la imagen panoramica
	xcen,ycen=xI-cen,yI-cen
	r=np.sqrt(xcen**2+ycen**2)
	fi=np.arctan2(ycen,xcen)
	
	## ahora a pasar a esfericas en el marco de referencia de la camara
	t=2*np.arctan(r/k) 
	Ct=np.cos(t) # coseno de theta
	St=np.sin(t) # seno de theta
	
	# para ahorrar cuentas y notacion, saco senos y cosenos una sola vez
	ca=np.cos(alfa)
	cb=np.cos(beta)
	cg=np.cos(gamma)
	sa=np.sin(alfa)
	sb=np.sin(beta)
	sg=np.sin(gamma)
	
	# he aqui la dichosa matriz
	T11,T12,T13=ca*cb,ca*sb*sg-sa*cg,ca*sb+sa*sg
	T21,T22,T23=sa*cb,sa*sb*sg+ca*cg,sa*sb*cg-ca*sg
	T31,T32,T33=-sb,cb*sg,cb*cg
	
	## salteandome las cartesianas paso directamente al mapa
	Rho = -Pz/(St*(T31*np.cos(fi)+T32*np.sin(fi))+Ct*T33)
	xM = Rho*(St*(T11*np.cos(fi)+T12*np.sin(fi))+T13*Ct)+Px
	yM = Rho*(St*(T21*np.cos(fi)+T22*np.sin(fi))+T23*Ct)+Py
	
	return xM,yM


# devuelve el error relativo pesado en distancia
def VCA2Earth_RelErr(prm,XX):
	# hago el forward de la transformacion
	XM_estimate = ForwardVCA2Map(prm,XX[0],XX[1])
	
	#distancia de la posicion que deberia dar a la camara
	d = np.sqrt(	(XX[2]-prm['Px'].value)**2 + \
			(XX[3]-prm['Py'].value)**2 + \
			prm['Pz'].value**2)
	
	# distancia de la posicion deseada y la obtenida
	e=np.sqrt((XM_estimate[0]-XX[2])**2+(XM_estimate[1]-XX[3])**2)
	
	# el error esta pesado segun la cercania la camara
	return e/d


#devuelve los errores relativo a la distancia y 'normal
def VCA2Earth_err(prm,XX):
	# hago el forward de la transformacion
	XM_estimate = ForwardVCA2Map(prm,XX[0],XX[1])
	
	#distancia de la posicion que deberia dar a la camara
	d = np.sqrt(	(XX[2]-prm['Px'].value)**2 + \
			(XX[3]-prm['Py'].value)**2 + \
			prm['Pz'].value**2)
	
	# distancia de la posicion deseada y la obtenida
	e=np.sqrt((XM_estimate[0]-XX[2])**2+(XM_estimate[1]-XX[3])**2)
	
	# el error esta pesado segun la cercania la camara
	return np.array([e,e/d])


#devuelve los errores totales para una lista de puntos
def VCA2Earth_ERR(prm,XX):
	E=np.array([0.0,0.0]) # donde acumular errores totales, normal y relativo
#	print XX.T
	for xx in XX.T:
#		print xx
		E+=VCA2Earth_err(prm,xx) # acumulo para este punto solo
	
	return E #retorno el total


# ponemos valores iniciales a las constantes (a ojo)
prm=Parameters()
prm.add('cen',value=960,vary=False)
prm.add('k',value=952.16,vary=False)
prm.add('gamma',value=np.pi) # repecto a x
prm.add('beta',value=0.0) # respecto a y
prm.add('alfa',value=np.pi/2) # respecto a z
prm.add('Px',value=290)
prm.add('Py',value=260)
prm.add('Pz',value=30)

# Guardar parametros
with open('parametrosiniciales.pkl', 'wb') as f:
	pickle.dump(prm, f)

# ahora probamos con parametros trivales
prm.add('cen',value=960,vary=False)
prm.add('k',value=952.16,vary=False)
prm.add('gamma',value=np.pi) # repecto a x, pone a la camara mirando para abajo
prm.add('beta',value=0.0) # respecto a y
prm.add('alfa',value=0.0) # respecto a z
prm.add('Px',value=324) # en el centro del mapa
prm.add('Py',value=241)
prm.add('Pz',value=10) # a altura arbitraria

# Guardar parametros iniciales oara condiciones 'triviales'
with open('parametroshard.pkl', 'wb') as f:
	pickle.dump(prm, f)



# para copiar los parametros
