clc, clear all, close all

Rept=10;
punto=1;
rng('default');
indixes = [];
load('DatosSeleccion.mat');
NumMuestras=size(X,1); 

NumClases=length(unique(Y)); %%% Se determina el número de clases del problema.

for fold=1:Rept

    %%% Se hace la partición de las muestras %%%
    %%%      de entrenamiento y prueba       %%%

    particion=cvpartition(NumMuestras,'Kfold',Rept);
    indices=particion.training(fold);
    Xtrain=X(particion.training(fold),:);
    Xtest=X(particion.test(fold),:);
    Ytrain=Y(particion.training(fold));
    Ytest=Y(particion.test(fold));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%% Entrenamiento de los modelos. Recuerde que es un modelo por cada clase. %%%


    NumArboles=15;
    %indice = SelectionGA(@FitnessSelection, Xtrain, Ytrain);
    indice = [1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0];
    Modelo=entrenarFOREST(NumArboles,Xtrain(:,indice==1),Ytrain);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%% Validación de los modelos. %%%

    Yest=testFOREST(Modelo,Xtest(:,indice==1));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    error=sum(Ytest ~= Yest)/length ( Yest) ;
    EficienciaTest(fold) = 1 - error;
    %indixes(fold,:) = indice;
end

Eficiencia = mean(EficienciaTest);
IC = std(EficienciaTest);

%Eficiencia1 = mean(EficienciaTest1);
%IC1 = std(EficienciaTest1);

Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
disp(Texto);

%%% Fin punto Random Forest %%%
    
