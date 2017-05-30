clc, clear all, close all

Rept=10;
punto=1;
indixes = [];
load('HOGFeatures.mat');
X = caracteristicas;
load('clases.mat');
Y = clases;
NumMuestras=size(X,1); 
boxConstraint = 10;

rng('default');
particion=cvpartition(NumMuestras,'Kfold',Rept);

NumClases=length(unique(Y)); %%% Se determina el número de clases del problema.
indexes = [];
%for tipo = {'linear', 'quadratic','diagLinear', 'diagQuadratic', 'mahalanobis'}

for fold=1:Rept

    %%% Se hace la partición de las muestras %%%
    %%%      de entrenamiento y prueba       %%%
    indices=particion.training(fold);
    Xtrain=X(particion.training(fold),:);
    [vectorPropio,~,valorPropio] = pca(Xtrain);
    newX = [];
    porcentaje = 0;
    x=0;
    while porcentaje<0.9
        x=x+1;
        porcentaje = porcentaje + (valorPropio(x)/sum(valorPropio));
    end
    Xtrain=X(particion.training(fold),:)*vectorPropio(:,1:x);
    Xtest=X(particion.test(fold),:)*vectorPropio(:,1:x);
    Ytrain=Y(particion.training(fold));
    Ytest=Y(particion.test(fold));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Normalización %%%

    [Xtrain,mu,sigma]=zscore(Xtrain);
    Xtest=(Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

    %%%%%%%%%%%%%%%%%%%%%
    %'linear', 'quadratic','diagLinear', 'diagQuadratic', or 'mahalanobis'
    %[clasificiacion] = classify(Xtest, Xtrain, Ytrain,tipo{1});
    [clasificiacion] = classify(Xtest, Xtrain, Ytrain,'linear');
    %%% Se encuentra la eficiencia y el error de clasificaciï¿½n %%%

    MatrizConfusion=zeros(11,11);
    for i=1:size(Xtest,1)
        MatrizConfusion(clasificiacion(i),Ytest(i))=MatrizConfusion(clasificiacion(i),Ytest(i)) + 1;
    end
    EficienciaTest(fold)=sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));

    indexes = [indexes, x];
end
    
Eficiencia = mean(EficienciaTest);
IC = std(EficienciaTest);

Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
disp(Texto);
%%% Fin punto Random Forest %%%
%end   
