# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 18:24:31 2014

@author: damzst
"""

import numpy as np
import cv2

# Function definitions are here
def showImage(IMAGE,WIN_NAME,WIN_HEIGHT,WIN_WIDTH):
    "Show Images on a pop-up window that is destroyed when pressing a key"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, WIN_HEIGHT, WIN_WIDTH) 
    cv2.imshow(WIN_NAME,IMAGE)
    cv2.waitKey(0)
    
    # IDK why, but destroyAllWindows ain't working, at least for me
    cv2.destroyAllWindows()
    # I found a not so optimal solution: call imshow again...
    cv2.imshow(WIN_NAME,IMAGE)
    return

def UIMr(l,m,theta):
    "Unified Model Imaging r(theta)"
    r = (l+m)*np.sin(theta)/(l+np.cos(theta))
    return r
    
def UIMtheta(l,m,r):
    "Unified Model Imaging theta(r)"
    r2 = r**2
    lm2 = (l+m)**2
    theta = np.arccos((l+m)*np.sqrt(r2*(1-l**2)+lm2-l*r2)/(r2+lm2))
    return theta
          
def interp2(A1,A2,A3,B1,B2):
    "2D Interpolation"
    # Get input size
#    height, width = A3.shape
    height, width = np.float64(A3.shape)
    # Get output size
#    heighto, widtho = B1.shape
    heighto, widtho = np.float64(B1.shape)
    # Flatten input arrays, just in case...
    A1 = A1.flatten('F')
    A2 = A2.flatten('F')
    A3 = A3.flatten('F')
    B1 = B1.flatten('F')
    B2 = B2.flatten('F')
    # Compute interpolation parameters    
    s = ((B1-A1[0])/(A1[-1]-A1[0]))*(width-1)
    t = ((B2-A2[0])/(A2[-1]-A2[0]))*(height-1)
#    s = 1+(B1-A1[0])/(A1[-1]-A1[0])*(width-1)
#    t = 1+(B2-A2[0])/(A2[-1]-A2[0])*(height-1)
    # Compute interpolation parameters pruebas
#    s = (B1-A1[0])/(A1[-1]-A1[0])*(width-1)
#    t = (B2-A2[0])/(A2[-1]-A2[0])*(height-1)

    print np.min(s),np.max(s),np.min(t),np.max(t)
    
    # Check for out of range values of s and t and set to 1
#    sout = np.nonzero(np.logical_or((s<1),(s>width)))
#    s[sout] = 1
#    tout = np.nonzero(np.logical_or((t<1),(t>width)))
#    t[tout] = 1
    # Check for out of range values of s and t and set to 0
#    sout = np.nonzero(np.logical_or((s<0),(s>width)))
#    s[sout] = 0
    s[s<0]=0; s[s>width]=0
    t[t<0]=0; t[t>width]=0
    # Matrix element indexing
    ndx = np.floor(t)+np.floor(s-1)*height
    ndx = np.intp(ndx)
    print ndx.shape, height, width, np.max(ndx)
    # Compute interpolation parameters
    s[:] = s-np.floor(s)
    t[:] = t-np.floor(t)
    np.disp(str(t))
    onemt = 1-t
    B3 = (A3[ndx-1]*onemt+A3[ndx]*t)*(1-s)+(A3[ndx+int(height)-1]*onemt+A3[ndx+int(height)])*s    
#    B3 = (A3[ndx]*onemt+A3[ndx+1]*t)*(1-s)+(A3[ndx+height]*onemt+A3[ndx+height+1])*s
    B3 = B3.reshape((heighto, widtho),order='F')   
    B3[B3<0]=0.
    B3[B3>255]=255.
    return B3


def findMR(u,v,u0,v0,l,m,tipo):
    "Find the rotation matrix"
    # Spherical coordinates corresponding to the point [u,v]
    phi = np.arctan2(v-v0,u-u0)
    r = np.sqrt((u-u0)**2+(v-v0)**2)
    theta = UIMtheta(l,m,r)
    
    # Vector coordinates
    P = np.array([np.sin(theta)*np.cos(phi),
                  np.sin(theta)*np.sin(phi),
                 -np.cos(theta)])    
    Pz = np.array([0,0,-1])
    
    # Versor of rotation
    Pk = np.cross(P,Pz)
    Pk = Pk/np.sqrt(np.sum(Pk**2))
    
    # Rotation Matrix
    c = np.cos(theta)
    s = np.sin(theta)
    v = 1-c
    
    kx = Pk[0]
    ky = Pk[1]
    kz = Pk[2]
    
    MR = np.array([[kx*kx*v+c, kx*ky*v-kz*s, kx*kz*v+ky*s],
                   [kx*ky*v+kz*s, ky*ky*v+c, ky*kz*v-kx*s],
                   [kx*kz*v-ky*s, ky*kz*v+kx*s, kz*kz*v+c]])
    
    if tipo == 'techo':
        beta = phi+np.pi/2
        cb = np.cos(beta)
        sb = np.sin(beta)
        Rz = np.array([[cb,-sb,0],[sb,cb,0],[0,0,1]])
        MR = np.dot(MR,Rz)
        
    return MR

#def rotsph(sph,MR):
#    "Rotate the spherical projection following MR transformation"
#    # Source image dimensions    
#    height, width = np.float64(sph.shape)
#    # Add grey array to cover whole sphere, not just the south
#    sph = np.vstack([np.ones(sph.shape)*.3,sph])
#    # Theta spans [0,pi/2][rad]
#    theta_range = np.linspace(0,np.pi/2,height) 
#    # Phi spans [-pi,pi][rad]
#    phi_range = np.linspace(-np.pi,np.pi,width)
#    # Build the plaid matrices
#    Phi, Theta = np.meshgrid(phi_range, theta_range)
#    # Convert the spherical coordinates to Cartesian
#    r = np.sin(Theta)
#    x = r*np.cos(Phi)
#    y = r*np.sin(Phi)
#    z = np.cos(Theta)
#    # Convert to 3xN format
#    p = np.transpose(np.hstack([x.flatten('F'),y.flatten('F'),z.flatten('F')]))
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
#    sph = interp2(np.linspace(0,heigth-1),np.linspace(0,width-1),sph,cx,cy)
#    return sph

# Load an image taken with VIVOTEK FE8172 from Oliva's balcony
src = cv2.imread('IM1.jpg')

# Show image
#showImage(src,'Fisheye Image', 480, 480)

# Get image size:
#height, width, depth = src.shape
height, width, depth = np.float64(src.shape)

# Principal point:
#u0 = width/2
#v0 = height/2
u0 = np.float64(width/2)
v0 = np.float64(height/2)

# Unified Imaging Model parameters for our camera:
#l = 1
#m = 952
l = np.float64(1)
m = np.float64(952)

# Coordinates of source image points:
#Ui, Vi = np.mgrid[0:width,0:height]
Ui, Vi = np.meshgrid(np.r_[0:width],np.r_[0:height])
#Ui, Vi = np.float64(np.meshgrid(np.r_[0:width],np.r_[0:height]))


# I define the coordinates of points on the spherical projection:
n = 1000
theta_range = np.linspace(0,np.pi/2,n) #[rad] De 0 a pi/2 (Medido desde abajo)
phi_range = np.linspace(-np.pi,np.pi,n)  #[rad] De -pi a pi
Phi, Theta = np.meshgrid(phi_range, theta_range)

# I calculate a grid that fits those angles
r = UIMr(l,m,Theta)
U = r*np.cos(Phi)+u0
V = r*np.sin(Phi)+v0

# Later, I project the image to an sphere:
sph = np.uint8(np.empty((n,n,3)))
sphf = np.float64(np.empty((n,n,3)))

for i in range(0,3):
    sphf[:,:,i] = np.float64(interp2(Ui,Vi,src[:,:,i],U,V))

# Show spherical image
#showImage(sph,'Spherical Image', 500, 500)

# Rotation of the spherical projection, moving the south to the point [u,v]
u = 1200
v = 1457
#MR = findMR(u,v,u0,v0,l,m,'techo')


phi = np.arctan2(v-v0,u-u0)
r = np.sqrt((u-u0)**2+(v-v0)**2)
theta = UIMtheta(l,m,r)

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
