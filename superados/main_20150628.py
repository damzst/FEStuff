# Imported libraries
import numpy as np
import cv2
import fun

# Load an image taken with VIVOTEK FE8172
src = cv2.imread('./resources/IM2.png')

# Get image size:
height, width, depth = np.float64(src.shape)

# Principal point:
u0 = np.float64(width/2)
v0 = np.float64(height/2)

# Unified Imaging Model parameters for our camera:
l = np.float64(1)
m = np.float64(952)

# Coordinates of source image points:
Ui, Vi = np.meshgrid(np.r_[0:width],np.r_[0:height])

# I define the coordinates of points on the spherical projection:
n = 1000
theta_range = np.linspace(0,np.pi/2,n) #[rad] De 0 a pi/2 (Medido desde abajo)
phi_range = np.linspace(-np.pi,np.pi,n)  #[rad] De -pi a pi
Phi, Theta = np.meshgrid(phi_range, theta_range)

# I calculate a grid that fits those angles
r = fun.UIMr(l,m,Theta)
U = r*np.cos(Phi)+u0
V = r*np.sin(Phi)+v0

# Later, I project the image to an sphere:
sphf = np.float64(np.empty((n,n,3)))

for i in range(3):
  sphf[:,:,i] = np.float64(fun.interp2(Ui,Vi,src[:,:,i],U,V))

# Show spherical image
cv2.imshow('Spherical Image',np.uint8(sphf))
cv2.waitKey(0)

# Rotation of the spherical projection, moving the south to the point [u,v]
u = 1200
v = 1457
#MR = findMR(u,v,u0,v0,l,m,'techo')


phi = np.arctan2(v-v0,u-u0)
r = np.sqrt((u-u0)**2+(v-v0)**2)
theta = fun.UIMtheta(l,m,r)

# Vector coordinates
#P = np.array([np.sin(theta)*np.cos(phi),
#              np.sin(theta)*np.sin(phi),
#             -np.cos(theta)])    
#Pz = np.array([0,0,-1])
#
## Versor of rotation
#Pk = np.cross(P,Pz)
#Pk = Pk/np.sqrt(np.sum(Pk**2))
#
## Rotation Matrix
#c = np.cos(theta)
#s = np.sin(theta)
#v = 1-c
#
#kx = Pk[0]
#ky = Pk[1]
#kz = Pk[2]
#
#MR = np.array([[kx*kx*v+c, kx*ky*v-kz*s, kx*kz*v+ky*s],
#               [kx*ky*v+kz*s, ky*ky*v+c, ky*kz*v-kx*s],
#               [kx*kz*v-ky*s, ky*kz*v+kx*s, kz*kz*v+c]])
#
#beta = phi+np.pi/2
#cb = np.cos(beta)
#sb = np.sin(beta)
#Rz = np.array([[cb,-sb,0],[sb,cb,0],[0,0,1]])
#MR = np.dot(MR,Rz)
#
#
#
#
#
#
#for i in range(0,3):
##    sph[:,:,i] = rotsph(sph[:,:,i],MR)
#    aux = np.vstack([np.ones(sphf[:,:,i].shape)*.3,sphf[:,:,i]])
#    # Convert the spherical coordinates to Cartesian
#    r = np.sin(Theta)
#    x = r*np.cos(Phi)
#    y = r*np.sin(Phi)
#    z = np.cos(Theta)
#    # Convert to 3xN format
#    p = np.vstack([x.flatten('F'),y.flatten('F'),z.flatten('F')])
#    # Transform points    
#    p = np.dot(MR,p)
#    # Reshape vectors
#    x = p[0,:].reshape(x.shape,order='F') 
#    y = p[1,:].reshape(x.shape,order='F')
#    z = p[2,:].reshape(x.shape,order='F')  
#    # Convert back to spherical coordinates
#    r = np.sqrt(x**2+y**2)
#    r[r>1] = 1
#    # Asin is multiple valued over the interval [0,pi]
#    nTheta = np.arcsin(r)
#    nTheta[z<0] = 0
#    nPhi = np.arctan2(y,x)
#    cx = (nPhi+np.pi)/(2*np.pi)*width
#    cy = nTheta/(np.pi/2)*height
#    # Warp the image
#    sphf[:,:,i] = interp2(np.linspace(0,height-1),np.linspace(0,width-1),aux,cx,cy)
#
#sph = np.uint8(sphf)
## Show spherical image
#showImage(sph,'Rotated Spherical Image', 500, 500)

# Field of view chosen [deg]
#fov = 120
#
## Size of the image wanted [pixels of side]
#W = 1500
#
## Unified Imaging Model parameters for this type of virtual camera
#mp = W/2/np.tan(fov/2*np.pi/180)
#lp = 0
#
## Principal point of this new image
#u0p = W/2 
#v0p = W/2
#
## Coordinates of this new image points
#Uo, Vo = np.meshgrid(np.r_[0:W],np.r_[0:W])
#
## Polar coordinates from the grid
#r = np.sqrt((Uo-u0p)**2 + (Vo-v0p)**2)
#phi = np.arctan2((Vo-v0p), (Uo-u0p))
#
## Spherical coordinates
#Phi_o = phi
#Theta_o = UIMtheta(lp,mp,r)
#
## Interpolation of images
#ptzv = np.uint8(np.empty((W,W,3)))
#
#for i in range(0,3):
#    ptzv[:,:,i] = np.uint8(interp2(Phi,Theta,sph[:,:,i],Phi_o,Theta_o))
#
## Show perspective image
#showImage(ptzv,'Virtual PTZ Image', 500, 500)
