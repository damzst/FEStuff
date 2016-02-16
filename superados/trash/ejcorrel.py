from numpy import arange, array, convolve,shape,argmax,correlate,abs

# Define toy vector arrays
rho1=array([0,1,0,0,0,0,0,0,0])
rho2=array([0,0,0,0,0,0,1,0,0])

# Length of arrays
#n=shape(rho1)[0]
n = len(rho1)

# Convolution of one respect to the other
c1=correlate(rho1,rho2,'full')
c2=correlate(rho2,rho1,'full')

# Displacement between signals
DT=abs(argmax(c1)-argmax(c2))/2

print rho1
print rho2
print 'correl 1',c1
print 'correl 2',c2
print 'desplazamiento temporal',DT

# The size of correlate output is [len(rho1)+len(rho2)]
DT1 = argmax(c1)-(n-1)

