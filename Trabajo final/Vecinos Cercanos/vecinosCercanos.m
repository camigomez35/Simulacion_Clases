function Yesti = vecinosCercanos(Xval,Xent,Yent,k,tipo)

    %%% El parametro 'tipo' es el tipo de problema que se va a resolver
    
    %%% La función debe retornar el valor de predicción Yesti para cada una de 
    %%% las muestras en Xval. Por esa razón Yesti se inicializa como un vectores 
    %%% de ceros, de dimensión M.

    N=size(Xent,1);
    M=size(Xval,1);
    
    Yesti=zeros(M,1);
    dis=zeros(N,1);

    if strcmp(tipo,'class')
        
        for j=1:M
            %%% Complete el codigo %%%
            for i = 1:N
                prueba = Xent(i,:)-Xval(j,:);
                prueba = prueba.^2;                
                prueba = sum(prueba);
                dis(i) = sqrt(prueba);
            end
            [kVecinos, indice] = sort(dis);
            kVecinos = kVecinos(1:k);
            indice = indice(1:k);
            
            Yesti(j)= mode(Yent(indice));
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
        
        
    elseif strcmp(tipo,'regress')
        
        for j=1:M
            %%% Complete el codigo %%%
            for i = 1:N
                prueba = Xent(i,:)-Xval(j,:);
                prueba = prueba.^2;                
                prueba = sum(prueba);
                dis(i) = sqrt(prueba);
            end
            [kVecinos, indice] = sort(dis);
            kVecinos = kVecinos(1:k);
            indice = indice(1:k);
            suma=0;
            for i=1:k
                suma = Yent(indice(i))+suma;
            end
            Yesti(j) = (1/k)*suma;
			%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        end

        
    end

end