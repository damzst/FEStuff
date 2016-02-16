'''
Where all the functions are
'''

# Imported libraries
import numpy as np
import cv2

# Constants definition
# New - cpp translation

# Function definition
# New - cpp translation
def map_sph(N_SPH,FE_L,FE_M,FE_U0,FE_V0):
  sph_x = np.zeros((N_SPH,N_SPH))
  sph_y = np.zeros((N_SPH,N_SPH))
  for i in range(N_SPH):
    theta = i*np.pi/2/N_SPH
    r = UIMr(FE_L,FE_M,theta)
    for j in range(N_SPH):
      phi = (j*2-N_SPH)*np.pi/N_SPH
      sph_x[i,j] = r*np.cos(phi) + FE_U0
      sph_y[i,j] = r*np.sin(phi) + FE_V0
  return sph_x,sph_y



# Old
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
    height, width = A3.shape
    # Get output size
    heighto, widtho = B1.shape
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
    # Compute interpolation parameters
    s[:] = s-np.floor(s)
    t[:] = t-np.floor(t)
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


