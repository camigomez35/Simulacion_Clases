clear all, close all

Rept=10;
punto=1;
rng('default');
indixes = [];
if punto==1
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
        %indice = sequentialfs(@Criterio, Xtrain, Ytrain);
        indice = [1 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0];
        Modelo=entrenarFOREST(NumArboles,Xtrain(:,indice==1),Ytrain);
        Modelo1=entrenarFOREST(NumArboles,Xtrain,Ytrain);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
      
        %%% Validación de los modelos. %%%
        
        Yest=testFOREST(Modelo,Xtest(:,indice==1));
        Yest1=testFOREST(Modelo1,Xtest);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        error = sum(Ytest ~= Yest)/length ( Yest) ;
        error1 = sum(Ytest ~= Yest1)/length ( Yest1) ;
        EficienciaTest(fold) = 1 - error;
        EficienciaTest1(fold) = 1 - error1;
        indixes(fold,:) = indice;
    end
    
    Eficiencia = mean(EficienciaTest);
    IC = std(EficienciaTest);
    
    Eficiencia1 = mean(EficienciaTest1);
    IC1 = std(EficienciaTest1);
    
    Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    Texto1=['La eficiencia obtenida fue = ', num2str(Eficiencia1),' +- ',num2str(IC1)];
    disp(Texto);
    disp(Texto1);

    %%% Fin punto Random Forest %%%
    
end