clc
clear all
close all

load('HOGFeatures.mat');
X = caracteristicas;
load('clases.mat');
Y = clases;

Rept = 10;
NumMuestras = size(X,1);

errorVect = [];
efiVect = [];

%%% Se hace la partición entre los conjuntos de entrenamiento y prueba.
%%% Esta partición se hace forma aletoria %%%

%%% Partición de las muestras
rng('default');
particion=cvpartition(NumMuestras,'Kfold',Rept);

superY = [];

for k = [5]
    Texto=strcat('Numero de vecinos: ',{' '},num2str(k));
    disp(Texto);
    for fold=1: Rept
            %%% Se hace la partición de las muestras
            %%% de entrenamiento y prueba
            indices=particion.training(fold);
            Xtrain=X(particion.training(fold),:);
            Xtest=X(particion.test(fold),:);
            Ytrain=Y(particion.training(fold),:);
            Ytest=Y(particion.test(fold),:);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%% Normalización %%%

            [Xtrain,mu,sigma]=zscore(Xtrain);
            Xtest=normalizar(Xtest,mu,sigma);

            %%%%%%%%%%%%%%%%%%%%%
            Yest=vecinosCercanos(Xtest,Xtrain,Ytrain,k,'class'); 

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            MatrizConfusion = zeros(11,11);
            for i=1:size(Xtest,1)
                MatrizConfusion(Yest(i),Ytest(i)) = MatrizConfusion(Yest(i),Ytest(i)) + 1;
            end
            EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
    end

    Eficiencia = mean(EficienciaTest);
    IC = std(EficienciaTest);
    
    Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    disp(Texto);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%