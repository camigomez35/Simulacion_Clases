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

    [Xtrain,mu,sigma]=zscore(Xtrain);
    Xtest=(Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

    Ytrain1=Ytrain;
    Ytrain2=Ytrain;
    Ytrain3=Ytrain;
    Ytrain4=Ytrain;
    Ytrain5=Ytrain;
    Ytrain6=Ytrain;
    Ytrain7=Ytrain;
    Ytrain8=Ytrain;
    Ytrain9=Ytrain;
    Ytrain10=Ytrain;
    Ytrain11=Ytrain;

    tic
    Ytrain1(Ytrain==1)=1;
    Ytrain1(Ytrain~=1)=-1;
    Modelo1=trainlssvm({Xtrain,Ytrain1,'c',boxConstraint,[], 'lin_kernel'});

    Ytrain2(Ytrain==2)=1;
    Ytrain2(Ytrain~=2)=-1;
    Modelo2=trainlssvm({Xtrain,Ytrain2,'c',boxConstraint,[], 'lin_kernel'});

    Ytrain3(Ytrain==3)=1;
    Ytrain3(Ytrain~=3)=-1;
    Modelo3=trainlssvm({Xtrain,Ytrain3,'c',boxConstraint,[], 'lin_kernel'});

    Ytrain4(Ytrain==4)=1;
    Ytrain4(Ytrain~=4)=-1;
    Modelo4=trainlssvm({Xtrain,Ytrain4,'c',boxConstraint,[], 'lin_kernel'});

    Ytrain5(Ytrain==5)=1;
    Ytrain5(Ytrain~=5)=-1;
    Modelo5=trainlssvm({Xtrain,Ytrain5,'c',boxConstraint,[], 'lin_kernel'});

    Ytrain6(Ytrain==3)=1;
    Ytrain6(Ytrain~=3)=-1;
    Modelo6=trainlssvm({Xtrain,Ytrain6,'c',boxConstraint,[], 'lin_kernel'});

    Ytrain7(Ytrain==7)=1;
    Ytrain7(Ytrain~=7)=-1;
    Modelo7=trainlssvm({Xtrain,Ytrain7,'c',boxConstraint,[], 'lin_kernel'});

    Ytrain8(Ytrain==8)=1;
    Ytrain8(Ytrain~=8)=-1;
    Modelo8=trainlssvm({Xtrain,Ytrain8,'c',boxConstraint,[], 'lin_kernel'});


    Ytrain9(Ytrain==9)=1;
    Ytrain9(Ytrain~=9)=-1;
    Modelo9=trainlssvm({Xtrain,Ytrain9,'c',boxConstraint,[], 'lin_kernel'});

    Ytrain10(Ytrain==10)=1;
    Ytrain10(Ytrain~=10)=-1;
    Modelo10=trainlssvm({Xtrain,Ytrain10,'c',boxConstraint,[], 'lin_kernel'});

    Ytrain11(Ytrain==11)=1;
    Ytrain11(Ytrain~=11)=-1;
    Modelo11=trainlssvm({Xtrain,Ytrain11,'c',boxConstraint,[], 'lin_kernel'});
    toc
    
    [Yest1, salida1Sim]=simlssvm(Modelo1,Xtest);
    [Yest2, salida2Sim]=simlssvm(Modelo2,Xtest);
    [Yest3, salida3Sim]=simlssvm(Modelo3,Xtest);
    [Yest4, salida4Sim]=simlssvm(Modelo4,Xtest);
    [Yest5, salida5Sim]=simlssvm(Modelo5,Xtest);
    [Yest6, salida6Sim]=simlssvm(Modelo6,Xtest);
    [Yest7, salida7Sim]=simlssvm(Modelo7,Xtest);
    [Yest8, salida8Sim]=simlssvm(Modelo8,Xtest);
    [Yest9, salida9Sim]=simlssvm(Modelo9,Xtest);
    [Yest10, salida10Sim]=simlssvm(Modelo10,Xtest);
    [Yest11, salida11Sim]=simlssvm(Modelo11,Xtest);

    aux=[salida1Sim,salida2Sim,salida3Sim,salida4Sim,salida5Sim,salida6Sim,salida7Sim,salida8Sim,salida9Sim,salida10Sim,salida11Sim];
    [~,Yest]=max(aux,[],2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    MatrizConfusion = zeros(11,11);
    for i=1:size(Xtest,1)
        MatrizConfusion(Yest(i),Ytest(i)) = MatrizConfusion(Yest(i),Ytest(i)) + 1;
    end
    EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
    indexes = [indexes, x];
end
    
Eficiencia = mean(EficienciaTest);
IC = std(EficienciaTest);

Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
disp(Texto);
%%% Fin punto Random Forest %%%
    
