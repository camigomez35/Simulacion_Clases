clc
close all
clear all

eta=0.1;     %%% Taza de aprendizaje
grado=2;     %%% Grado del polinomio

rng('default');
% Tipo=input('Ingrese 1 para regresion y 2 para clasificacion: ');

%%%
Tipo = 1;
%%%

if Tipo==1
    
    %%% Se crean los datos de forma aleatoria %%%
    
    X1=linspace(-20,20,500);
    X2=linspace(-50,50,500);
    X=[X1',X2'];    
    X=zscore(X);
    Y=2*X.^3 + 3*X.^2 - 4*X + 6;
    Y=min(abs(Y),[],2) + max(abs(Y),[],2).*0.2.*randn(500,1);
    X=[X1',X2'];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Se cambia el grado del polinomio %%%
            
    X=potenciaPolinomio(X,grado);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

    %%% Se hace la particion entre los conjuntos de entrenamiento y prueba.
    %%% Esta particion se hace forma aletoria %%%
    
    rng('default');
    ind=randperm(500); %%% Se seleccionan los indices de forma aleatoria
    
    Xtrain=X(ind(1:450),:);
    Xtest=X(ind(451:end),:);
    Ytrain=Y(ind(1:450),:);
    Ytest=Y(ind(451:end),:);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Normalizacion %%%
    
    [Xtrain,mu,sigma]=zscore(Xtrain);
    Xtest=normalizar(Xtest,mu,sigma);
    
    %%%%%%%%%%%%%%%%%%%%%
    
    %%% Se extienden las matrices %%%
    
    Xtrain=[Xtrain,ones(450,1)];
    Xtest=[Xtest,ones(50,1)];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Se aplica la regresion multiple %%%
    
    W=regresionMultiple(Xtrain,Ytrain,eta); %%% Se optienen los W coeficientes del polinomio
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Se encuentra el error cuadratico medio %%%
    
    Yesti=(W'*Xtest')';
    ECM=(sum((Yesti-Ytest).^2))/length(Ytest);
    
%     Texto=strcat('El Error cuadr�tico medio en prueba es: ',{' '},num2str(ECM));
    Texto=['El Error cuadr�tico medio en prueba es: ',num2str(ECM)];
    disp(Texto);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
elseif Tipo==2
    
    %%% Se crean los datos de forma aleatoria %%%
    
    rng('default');
    
    a=(2*pi).*rand(250,1);
    r=sqrt(rand(250,1))+1.5;
    carac1=(4*r).*cos(a)+0;
    carac2=(3*r).*sin(a)+0;
    X1=[carac1,carac2];
    figure; plot(X1(:,1),X1(:,2),'xr');
    hold on
    
    a=(2*pi).*rand(250,1);
    r=sqrt(rand(250,1)) + rand(250,1);
    carac1=(3*r).*cos(a)+0;
    carac2=(2*r).*sin(a)+0;
    X2=[carac1,carac2];
    plot(X2(:,1),X2(:,2),'xb');

    X=[X1;X2];
    
    Y=[ones(250,1);zeros(250,1)];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Se cambia el grado del polinomio %%%

    X=[X1;X2];
    X=potenciaPolinomio(X,grado);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %%% Se hace la partici�n entre los conjuntos de entrenamiento y prueba.
    %%% Esta partici�n se hace forma aletoria %%%
    
    rng('default');
    ind=randperm(500); %%% Se seleccionan los indices de forma aleatoria
    
    Xtrain=X(ind(1:450),:);
    Xtest=X(ind(451:end),:);
    Ytrain=Y(ind(1:450),:);
    Ytest=Y(ind(451:end),:);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Normalizaci�n %%%
    
    [Xtrain,mu,sigma]=zscore(Xtrain);
    Xtest=normalizar(Xtest,mu,sigma);
    
    %%%%%%%%%%%%%%%%%%%%%
    
    
    
    %%% Se extienden las matrices %%%
    
    Xtrain=[Xtrain,ones(450,1)];
    Xtest=[Xtest,ones(50,1)];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    [clasificiacion] = classify(Xtest(:,1:4), Xtrain(:,1:4), Y(1:450,1),'quadratic');
    
    
    
    
    %%% Se aplica la regresi�n logistica %%%
    
    W=regresionLogistica(Xtrain,Ytrain,eta); %%% Se optienen los W coeficientes del polinomio
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Se encuentra la eficiencia y el error de clasificaci�n %%%
   
    Yesti=(W'*Xtest')';
    Yesti(Yesti>=0)=1;
    Yesti(Yesti<0)=0;
    
    Eficiencia=(sum(clasificiacion==Ytest))/length(Ytest);
    Error=1-Eficiencia;
    
%     Texto=strcat('La eficiencia en prueba es: ',{' '},num2str(Eficiencia));
    Texto=['La eficiencia en prueba es: ',num2str(Eficiencia)];
    disp(Texto);
%     Texto=strcat('El error de clasificaci�n en prueba es: ',{' '},num2str(Error));
    Texto=['El error de clasificaci�n en prueba es: ',num2str(Error)];
    disp(Texto);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Eficiencia=(sum(Yesti==Ytest))/length(Ytest);
    Error=1-Eficiencia;
    
%     Texto=strcat('La eficiencia en prueba es: ',{' '},num2str(Eficiencia));
    Texto=['La eficiencia en prueba es: ',num2str(Eficiencia)];
    disp(Texto);
%     Texto=strcat('El error de clasificaci�n en prueba es: ',{' '},num2str(Error));
    Texto=['El error de clasificaci�n en prueba es: ',num2str(Error)];
    disp(Texto);
end

