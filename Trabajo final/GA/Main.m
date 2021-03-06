clc, clear all, close all

Rept=10;
punto=1;
rng('default');
indixes = [];
load('HOGFeatures.mat');
X = caracteristicas;
load('clases.mat');
Y = clases;

NumMuestras = size(X,1);
NumClases=length(unique(Y)); %%% Se determina el n�mero de clases del problema.
fold =3;
%for fold=1:Rept

    %%% Se hace la partici�n de las muestras %%%
    %%%      de entrenamiento y prueba       %%%

    particion=cvpartition(NumMuestras,'Kfold',Rept);
    indices=particion.training(fold);
    Xtrain=X(particion.training(fold),:);
    Xtest=X(particion.test(fold),:);
    Ytrain=Y(particion.training(fold));
    Ytest=Y(particion.test(fold));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%% Entrenamiento de los modelos. Recuerde que es un modelo por cada clase. %%%


    NumArboles=40;
    indice = SelectionGA(@FitnessSelection, Xtrain, Ytrain);
    Modelo=entrenarFOREST(NumArboles,Xtrain(:,indice==1),Ytrain);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%% Validaci�n de los modelos. %%%

    Yest=testFOREST(Modelo,Xtest(:,indice==1));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    MatrizConfusion=zeros(11,11);
    for i=1:size(Xtest,1)
        MatrizConfusion(Yest(i),Ytest(i))=MatrizConfusion(Yest(i),Ytest(i)) + 1;
    end
    EficienciaTest(fold)=sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));


%end 
%%% Se encuentra la eficiencia y el error de clasificaci�n %%%

Eficiencia = mean(EficienciaTest);
IC = std(EficienciaTest);
Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
disp(Texto);

