clc
clear all
close all

load('HOGFeatures.mat');
X = caracteristicas;
load('clases.mat');
Y = clases;

Rept = 10;
NumMuestras = size(X,1);
NumClases=length(unique(Y));

EficienciaTest=zeros(1,Rept);

%%% Se hace la partición entre los conjuntos de entrenamiento y prueba.
%%% Esta partición se hace forma aletoria %%%

rng('default');
particion=cvpartition(NumMuestras,'Kfold',Rept);

for NumArboles=[40]
    Texto=['Número de arboles = ', num2str(NumArboles)];
    disp(Texto);
    tiempo = [];
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

        [XtrainNormal,mu,sigma] = zscore(Xtrain);
        XtestNormal = (Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

        %%%%%%%%%%%%%%%%%%%%%
        
        Modelo=entrenarFOREST(NumArboles,Xtrain,Ytrain);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Validación de los modelos. %%%
        tiempo_inicio = cputime;
        Yest=testFOREST(Modelo,Xtest);
        total = cputime - tiempo_inicio;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tiempo = [tiempo, total];
        MatrizConfusion = zeros(NumClases,NumClases);
        for i=1:size(Xtest,1)
            MatrizConfusion(Yest(i),Ytest(i)) = MatrizConfusion(Yest(i),Ytest(i)) + 1;
        end
        EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
    end

    Eficiencia = mean(EficienciaTest);
    IC = std(EficienciaTest);
    Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    disp(Texto);
    disp(['Tiempo', num2str(mean(tiempo))]);
end