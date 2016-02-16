function MR = findMR(u,v,u0,v0,l,m,tipo)
% Se debe tener en cuenta que, como la funcion interp2 requiere que la
% primer malla ingresada (U1,V1) sea uniforme, la malla que se rotará no
% será la de la proyección original. La que se rotará será, en cambio, la
% malla sin intensidades (U2,V2)=(U1,V1).
% Ésta, una vez rotada, será interpolada con la malla original de la
% proyección I3=interp2(U1,V1,I1,U2,V2), estos valores de intensidad serán
% guardados en otra malla vacía pero con (U3,V3)=(U1,V1).

% Se obtienen phi y theta correspondientes a ese punto:
phi = atan2(v-v0,u-u0);
r = sqrt((u-u0)^2+(v-v0)^2);
theta = unified_theta([l,m],r);

% Coordenadas de los vectores:
P = [sin(theta)*cos(phi);...
     sin(theta)*sin(phi);...
     -cos(theta)];
Pz = [0 0 -1];

% El versor del eje alrededor del cual se quiere rotar es Pk:
Pk = cross(P,Pz);
Pk = Pk/sqrt(sum(Pk.^2)); % Se normaliza

% Se contruye la matriz de rotación MR, la cual rota alrededor del versor
% Pk, un ángulo alpha = theta.
alpha = theta;

c = cos(alpha); 
s = sin(alpha);
v = 1-c;

kx = Pk(1); 
ky = Pk(2);
kz = Pk(3);

MR(1,1) = kx*kx*v+c;    MR(1,2) = kx*ky*v-kz*s; MR(1,3) = kx*kz*v+ky*s;
MR(2,1) = kx*ky*v+kz*s; MR(2,2) = ky*ky*v+c;    MR(2,3) = ky*kz*v-kx*s;
MR(3,1) = kx*kz*v-ky*s; MR(3,2) = ky*kz*v+kx*s; MR(3,3) = kz*kz*v+c;

if strcmp(tipo,'techo')
% Para orientar la imagen en el caso de imágenes tomadas del techo, se 
% requiere girar la esfera alrededor de z un ángulo beta:
beta = phi+pi/2;
cp = cos(beta);
sp = sin(beta);
Rz = [cp,-sp,0;...
      sp, cp,0;...
      0,   0,1];
MR = MR * Rz;
end