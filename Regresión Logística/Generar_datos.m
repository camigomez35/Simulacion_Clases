%Limpiar 
clear all, close all, clc
%Generar valores
x1 = 2*randn(100,2)-2;
x2 = 0.5*rand(100,2)+3;
%Concatenar los datos en la misma matriz
x=[x1;x2]; 
y=[ones(100,1);zeros(100,1)];
figure(1);
plot(x(1:100,1),x(1:100,2),'xr');
hold on
plot(x(101:end,1),x(101:end,2),'xb');
hold off
%Llamar funcion
gradiente(x,y);