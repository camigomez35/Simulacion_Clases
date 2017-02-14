%Limpiar 
clear all, close all, clc
%Generar valores
x1 = 2*randn(100,2)-2;
x2 = 0.5*rand(100,2)+3;
%Concatenar los datos en la misma matriz
x=[x1;x2]; 
y=[ones(100,1);zeros(100,1)];
%Llamar funcion
gradiente(x,y);
