
clc
clear all
close all

Rept=10;

punto=input('Ingrese 1 para regresi�n � 2 para clasificaci�n: ');
boxConstraint=0.01;
gamma=0.01;

if punto==1
    
    %%% punto de regresi�n %%%
    
    load('DatosRegresion.mat');
    ECMTest=zeros(1,Rept);
    NumMuestras=size(X,1);
        
    for fold=1:Rept

        %%% Se hace la partici�n de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        
        rng('default');
        particion=cvpartition(NumMuestras,'Kfold',Rept);
        Xtrain=X(particion.training(fold),:);
        Xtest=X(particion.test(fold),:);
        Ytrain=Y1(particion.training(fold));
        Ytest=Y1(particion.test(fold));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Se normalizan los datos %%%

        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=(Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Entrenamiento del modelo. %%%

        Modelo=entrenarSVM(Xtrain,Ytrain,..........); %incluir par�metros adicionales, boxcontrain,gamma si aplica, etc...

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Validaci�n del modelo. %%%

        Yest=testSVM(Modelo,Xtest);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        ECMTest(fold)=(sum((Yest-Ytest).^2))/length(Ytest);
        
    end
    
    ECM = mean(ECMTest);
    IC = std(ECMTest);
    Texto=['El error cuadratico medio obtenida fue = ', num2str(ECM),' +- ',num2str(IC)];
    disp(Texto);

    %%% Fin punto de regresi�n %%%
    
elseif punto==2
    
    %%% punto clasificaci�n %%%
    
    load('DatosClasificacion.mat');
    NumClases=length(unique(Y)); %%% Se determina el n�mero de clases del problema.
    EficienciaTest=zeros(1,Rept);
    NumMuestras=size(X,1);
    
    for fold=1:Rept

        %%% Se hace la partici�n de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        
        rng('default');
        particion=cvpartition(NumMuestras,'Kfold',Rept);
        Xtrain=X(particion.training(fold),:);
        Xtest=X(particion.test(fold),:);
        Ytrain=Y(particion.training(fold),:);
        Ytest=Y(particion.test(fold));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Se normalizan los datos %%%

        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=(Xtest - repmat(mu,size(Xtest,1),1))./repmat(sigma,size(Xtest,1),1);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%% Entrenamiento de los modelos. Se usa la metodologia One vs All. %%%
        
        %%% Complete el codigo implimentando la estrategia One vs All.
		%%% Recuerde que debe de entrenar un modelo SVM para cada clase.
		%%% Solo debe de evaluar las muestras con conflicto.
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        MatrizConfusion=zeros(NumClases,NumClases);
        for i=1:size(Xtest,1)
            MatrizConfusion(Yest(i),Ytest(i))=MatrizConfusion(Yest(i),Ytest(i)) + 1;
        end
        EficienciaTest(fold)=sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        
    end
    
    Eficiencia = mean(EficienciaTest);
    IC = std(EficienciaTest);
    Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    disp(Texto);

    %%% Fin punto de clasificaci�n %%%

end



