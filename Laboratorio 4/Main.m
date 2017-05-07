clc
clear all
close all
%%% Maria Camila Gómez 
%%% Santiago Romero
Rept=10;
%%% Lectura de los datos y separación del conjunto de muestras y conjunto
%%% de salidas
Data = xlsread('ENB2012_data.xlsx');
X=Data(:,1:8);      %Muestras
Y=Data(:,9:end);    %Salidas
NumMuestras=size(X,1);

%%% Iteraciones sobre la malla de valores para el número de neuronas
for S = [1 16 25 50 75 100]
    Texto=['El número de neuronas a evaluar es ', num2str(S)];
    disp(Texto);
    
    %%% Iteraciones sobre las distintas cantidades de épocas
    for epochs = [100 400 800 1000]

        ecmVect = [];

        for fold=1: Rept

            %%% Se hace la partición de las muestras
            %%% de entrenamiento y prueba
            rng('default');
            particion=cvpartition(NumMuestras,'Kfold',Rept);
            indices=particion.training(fold);
            Xtrain=X(particion.training(fold),:);
            Xtest=X(particion.test(fold),:);
            Ytrain=Y(particion.training(fold),:);
            Ytest=Y(particion.test(fold),:);

            %%% Se normalizan los datos
            [XtrainNormal,mu,sigma] = zscore(Xtrain);
            XtestNormal = (Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);
            
            %%% Declaración, definición de épocas máximas y entrenamiento
            %%% de la red neuronal usando la función fitnet()
            net = fitnet(S);
            net.trainParam.epochs = epochs;
            net = train(net, Xtrain',Ytrain');
            
            %%% Evaluación de la red neuronal usando el conjunto de datos
            %%% de validación
            Yesti = net(Xtest');
            
            %%% Cálculo del error cuadrático medio
            ECM=(sum((Yesti'-Ytest).^2))/length(Ytest);
            ecmVect = [ecmVect, ECM];
            
        end

        %%% Media del error cuadrático medio para el conjunto de folds
        ECM = mean(ecmVect);
        Texto=['El error cuadrático medio fue: ',num2str(ECM), ' con ', num2str(epochs), ' epocas.'];
        disp(Texto);
    end
end