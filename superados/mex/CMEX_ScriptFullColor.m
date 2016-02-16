%% Mapeo de Fisheye -> Esfera -> Imagen Plana Color
% Usando CMEX.
clear all; close all; clc

mainfolder = '/home/damzst/Documents/MATLAB/14_1_tpf/Experimentos';
addpath(mainfolder,...
        [mainfolder,'/rvctools'],...
        [mainfolder,'/FisheyeMdls']);
run('startup_rvc')

%% Se prueba el mapeo de la imagen a una esfera
fisheye = iread('IM1.jpg', 'double');
figure('name','Imagen fisheye')
imshow(fisheye,'InitialMagnification',25)
% Dimensiones de la imagen:
width = size(fisheye,2); height = size(fisheye,1);
% Punto principal:
u0 = width/2; v0 = height/2;
% Parametros del modelo unificado:
l = 1; m = 952;
% Se define el dominio de los puntos de la imagen de entrada
[Ui,Vi] = meshgrid(1:width, 1:height);
% El dominio de los puntos de la imagen de salida cubre el hemisferio inf.
n = 1500;
theta_range = (0:n)/n*pi/2; %[rad] De 0 a pi/2 (Medido desde abajo)
phi_range = (-n:2:n)/n*pi;  %[rad] De -pi a pi
[Phi,Theta] = meshgrid(phi_range, theta_range);
% Se obtiene una malla de puntos en la imagen correspondientes a esos
% angulos:
r = unified([l,m],Theta);
U = r.*cos(Phi) + u0;
V = r.*sin(Phi) + v0;
% Despues se aplica el warp hacia una esfera:
spherical = zeros(n+1,n+1,3);
for i = 1:3
     spherical(:,:,i) = myinterp2(Ui, Vi, fisheye(:,:,i), U, V);
end
% Rotacion de la esfera, apunta al punto de la imagen [u,v]:
% u = 1200; v = 1457;
% MR = findMR(u,v,u0,v0,l,m,'techo');
% for i = 1:3
% spherical(:,:,i) = sphere_rotate(spherical(:,:,i),MR);
% end

% Se puede ver la imagen distorsionada obtenida en la esfera:
figure('name','Imagen en la esfera')
sphere_paint(spherical); view([-90 0])
title('Esfera vista desde el sur')
ylabel('Y'),xlabel('X')

%% Se prueba el mapeo de la imagen esferica a una imagen plana
% Campo de vision elegido:
fov = 120; %[deg]
% La imagen sera cuadrada y de lado:
W = 1500; %[pixels]
% Par�metros para este tipo de camara (perspectiva):
mp = W / 2 / tan(fov/2*pi/180); %[pixels]
lp = 0; %[pixels]
% Punto principal en el centro de la imagen:
u0p = W/2; v0p = W/2;
% Dominio de imagen de salida:
[Uo,Vo] = meshgrid(0:W-1, 0:W-1);
% Las coordenadas polares de cada punto de la imagen de salida:
r = sqrt((Uo-u0p).^2 + (Vo-v0p).^2);
phi = atan2((Vo-v0p), (Uo-u0p));
% Sus correspondientes coordenadas polares:
Phi_o = phi;
Theta_o = unified_theta([lp,mp],r);
% Se mapea entonces a la imagen plana:
perspective = zeros(W,W,3); 
for i = 1:3
perspective(:,:,i) = interp2(Phi, Theta, spherical(:,:,i), Phi_o, Theta_o);
end
% La imagen obtenida es entonces:
figure('name','Imagen en la proyecci�n plana')
imshow(perspective,'InitialMagnification',25)
axis on, ylabel('v [pixels]'),xlabel('u [pixels]')
title(['Imagen en la proyecci�n plana',' con FOV = ', sprintf('%.0f',fov),'�'],...
    'FontSize',12)

%% Se desea enmarcar la imagen del campo de visi�n en la imagen original
% Para lograr esto, se proyecta el marco desde la imagen plana final hacia
% la proyecci�n estereografica

% Primero, las coordenadas del marco:
% t = 30;
% s = W-t;
% w = 20;
% Uf = [        t:s, s*ones(1,s), fliplr(t:s), t*ones(1,s)];
% Vf = [t*ones(1,s),         t:s, s*ones(1,s), fliplr(t:s)];
% % La imagen tendr� cuadrados blancos de (2*w+1)x(2*w+1) pixels en la
% % posici�n de cada punto definido por (Uf,Vf):
% marco = zeros(W);
% for i=1:length(Uf)
%     marco(Vf(i)-w:Vf(i)+w,Uf(i)-w:Uf(i)+w) = 255;
% end
% % Se mapea entonces a la proyecci�n esf�rica:
% rm = unified([lp,mp],Theta);
% % Se acotan los valores de rm:
% ixrm = rm>2000;
% rm(ixrm) = 2000;
% % Se genera la malla de puntos a mapear:
% Um = rm.*cos(Phi) + u0p;
% Vm = rm.*sin(Phi) + v0p;
% marcosph = myinterp2(Uo, Vo, marco, Um, Vm);
% marcosph = sphere_rotate(marcosph,inv(MR));
% % Las coordenadas polares de cada punto de la imagen de salida:
% rfe = sqrt((Ui-u0).^2 + (Vi-v0).^2);
% phife = atan2((Vi-v0), (Ui-u0));
% % Sus correspondientes coordenadas polares:
% Phife = phife;
% Thetafe = unified_theta([l,m],rfe);
% % Se mapea entonces a la imagen plana:
% marcofe = interp2(Phi, Theta, marcosph, Phife, Thetafe);
% % Se acotan los valores:
% ix = isnan(marcofe);
% marcofe(ix) = 0;
% % Se superponen las im�genes:
% marcofemfe = ~marcofe.*fisheye(:,:,1);
% marcofemfe = cat(3,marcofemfe,~marcofe.*fisheye(:,:,2));
% marcofemfe = cat(3,marcofemfe,~marcofe.*fisheye(:,:,3));
% % La imagen obtenida es entonces:
% figure('name','Imagen enmarcada')
% imshow(marcofemfe,'InitialMagnification',25)
% axis on, ylabel('v [pixels]'),xlabel('u [pixels]')
% title('Imagen enmarcada en la proyeccion estereogr�fica')
% 
% %% Guardado de imagenes
% imwrite(perspective,'Plana.jpg','jpg')
% imwrite(marcofemfe,'Marco.jpg','jpg')
