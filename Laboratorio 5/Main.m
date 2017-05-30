clc, clear all, close all

Rept=10;

punto=input('Ingrese 1 para regresión ó 2 para clasificación: ');
boxConstraint=10;
gamma=100;

if punto==1
    % for boxConstraint = [0.01, 0.1, 1, 10, 100]
        %for gamma = [0.01, 0.1, 1, 10, 100]
            %%% punto de regresión %%%

            load('DatosRegresion.mat');
            ECMTest=zeros(1,Rept);
            NumMuestras=size(X,1);

            for fold=1:Rept

                    %%% Se hace la partición de las muestras %%%
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

                    Modelo=trainlssvm({Xtrain,Ytrain,'f',boxConstraint,gamma});
                    %plotlssvm(Modelo);
                    %incluir parámetros adicionales, boxcontraint,gamma si aplica, etc...

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    %%% Validación del modelo. %%%

                    Yest=simlssvm(Modelo,Xtest);

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    ECMTest(fold)=(sum((Yest-Ytest).^2))/length(Ytest);

            end

            ECM = mean(ECMTest);
            IC = std(ECMTest);
            Texto = ['Gamma = ', num2str(gamma), ' Box = ', num2str(boxConstraint)];
            disp(Texto);
            Texto=['El error cuadratico medio obtenida fue = ', num2str(ECM),' +- ',num2str(IC)];
            disp(Texto);

            %%% Fin punto de regresión %%%
         %end
    %end
elseif punto==2
    for boxConstraint = [0.01, 0.1, 1, 10, 100]
        %for gamma = [0.01, 0.1, 1, 10, 100]
    %%% punto clasificación %%%
    
    load('DatosClasificacion.mat');
    NumClases=length(unique(Y)); %%% Se determina el número de clases del problema.
    EficienciaTest=zeros(1,Rept);
    NumMuestras=size(X,1);
    
    for fold=1:Rept

        %%% Se hace la partición de las muestras %%%
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
        
        Ytrain1=Ytrain;
        Ytrain2=Ytrain;
        Ytrain3=Ytrain;
        
        Ytrain1(Ytrain==1)=1;
        Ytrain1(Ytrain~=1)=-1;
        Modelo1=trainlssvm({Xtrain,Ytrain1,'c',boxConstraint,[],'lin_kernel'});
        
        Ytrain2(Ytrain==2)=1;
        Ytrain2(Ytrain~=2)=-1;
        Modelo2=trainlssvm({Xtrain,Ytrain2,'c',boxConstraint,[],'lin_kernel'});
       
        Ytrain3(Ytrain==3)=1;
        Ytrain3(Ytrain~=3)=-1;
        Modelo3=trainlssvm({Xtrain,Ytrain3,'c',boxConstraint,[],'lin_kernel'});
        
         [Yest1, salida1Sim]=simlssvm(Modelo1,Xtest);
         [Yest2, salida2Sim]=simlssvm(Modelo2,Xtest);
         [Yest3, salida3Sim]=simlssvm(Modelo3,Xtest);
        
%        salida1=evaluarFuncioSVM(Modelo1.alpha, Modelo1.b, Modelo1.ytrain, Modelo1.xtrain, Xtest,gamma,'gauss');
%        salida2=evaluarFuncioSVM(Modelo2.alpha, Modelo2.b, Modelo2.ytrain, Modelo2.xtrain, Xtest,gamma,'gauss');
%        salida3=evaluarFuncioSVM(Modelo3.alpha, Modelo3.b, Modelo3.ytrain, Modelo3.xtrain, Xtest,gamma,'gauss');
         aux=[salida1Sim,salida2Sim,salida3Sim];
         [~,Yest]=max(aux,[],2);


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        
        
        
        MatrizConfusion=zeros(NumClases,NumClases);
        for i=1:size(Xtest,1)
            MatrizConfusion(Yest(i),Ytest(i))=MatrizConfusion(Yest(i),Ytest(i)) + 1;
        end
        EficienciaTest(fold)=sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        
    end
    
    Eficiencia = mean(EficienciaTest);
    IC = std(EficienciaTest);
    Texto = ['Gamma = ', num2str(gamma), ' Box = ', num2str(boxConstraint)];
    disp(Texto);
    Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    disp(Texto);

    %%% Fin punto de clasificación %%%
    end
   %end
end



