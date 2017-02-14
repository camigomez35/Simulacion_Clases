clear all, close all, clc
%variables pueden tomar cualquier valor
m = 'string';
%crear vector de 1 a 10
v = 1:10;
v
%crea vector de 1 a 10 con saltos de a 0.5
v = 1:0.5:10;
v
%vector fila
v2 = [2,3,5,9];
v2
%vector columna
v2 = [2;3;5;9];
v2
%Vector de unos de 3x3
ones(3)
exp(ans)
%remplazar valores en un vector
a(1,1,1)=10;
a(1,2,1)=10;
a
%tamaño de v
length(v)
m = ones(5);
[fil, col] = size(m);
%transpuesta
m2 = [2,3,4;6 4 5;9 3 4];
m2
m2'

M = randn(10)
M(3,5)
clc
M(3,5)
M(3,:)
M(:,3)
clc
M(3,:)=1;
M
clc
M(3,:)=rand(1,10);
M(3,:)=rand(10,1);
rand(10,1);
rand(10,1)
M(3,:)=ans;
clc
help mean
clc
mean(M)
mean(M,2)
clc
mean(M)
ind = mean(M)>0.28;
ind
M(:,ind)
M(:,mean(M)>0.28)
A = randn(2,3);
B = randn(4,3);
D = A.*A;
A
mifuncion(@mifuncion2,4)
f = @(x) x.^2;
f(3);
f([3,5]);
%Estructuras
E.Experimento(1).Nombre = 'name1';
E.Experimento(1).Matriz = randn(2,3);
E.Experimento(1).vector = randn(1,3);
E.Experimento(1).flags = randn(3,3)>0.3;
E.Experimento(2).Nombre = 'name2';
E.Experimento(2).Matriz = randn(2,3);
E2(1)=E;
E2(2)=E;

%otra cosa
x = linspace(-10,10,1000);
y = 2*x + 3*x.^2 - 4;
plot(x,y)
plot(x,y,'.')
plot(x,y)
hold on
y = x + 2*x.^2 - 6;
plot(x,y)
plot(x,y,'r')
plot(x,y,'r.')
plot(x,y,'r-')
plot(x(1:10),y(1:10),'r-')
plot(x(1:10),y(1:10),'r.-')
y = 2*x + 3*x.^2 - 4;

figure(2)
hold on
plot(x(1:10),y(1:10),'b.-')
legend('y1','y2')
hist(randn(1,1000))
hist(rand(1,1000))
f = @(x,y) exp(cos(sqrt(x.^2 + y.^2)));
d = -2*pi:0.1:2*pi;
[x,y] = meshgrid(d,d);
z = f(x,y);

figure(4)
surf(x,y,z)
shading interp
camlight(90,0)
alpha(0.8)
box on