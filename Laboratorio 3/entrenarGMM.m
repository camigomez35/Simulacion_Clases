function modelo = entrenarGMM(X,NumeroMezclas)

    inputDim=size(X,2);      %%%%% Numero de caracteristicas de las muestras
    modelo = gmm(inputDim, NumeroMezclas,'full');
    options = foptions;
    options(1)=0;
    %options(3)=0;
    options(5)=0;
    options(14)=50;
    modelo = gmminit(modelo, X, options);
    modelo = gmmem(modelo, X, options);
    
end