# -*- coding: utf-8 -*-
"""
ejemplo de como usar ForwardVCA2Map

@author: sarroyo

"""

import pickle
from Transform import ForwardVCA2Map

# load optimized parameters (best fit)
#prm=Parameters()
with open('parametros.pkl', 'rb') as f:
	prm = pickle.load(f)


x2,y2,=ForwardVCA2Map(prm,x,y) # con parametros optimos



