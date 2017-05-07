  function Yesti = ventanaParzen(Xval,Xent,Yent,h,tipo)

          %%% La función debe retornar el valor de predicción Yesti para cada una de 
	  %%% las muestras en Xval. Por esa razón Yesti se inicializa como un vectores 
	  %%% de ceros, de dimensión M.
  
      M=size(Xval,1);
      N=size(Xent,1);
      
      Yesti=zeros(M,1);

      if strcmp(tipo,'regress')
      
	  for j=1:M
	    %%% Complete el codigo %%%
            suma = 0;
            suma2=0;
            for i = 1:N
                prueba = Xval(j,:)-Xent(i,:);   
                prueba = sqrt(norm(prueba));
                prueba = prueba/h;
                prueba = gaussianKernel(prueba);
                a = prueba*Yent(i);
                suma2 = suma2 + prueba;
                suma = suma + a;
            end
           
            Yesti(j)= suma/suma2;
	    %%%%%%%%%%%%%%%%%%%%%%%%%%
	    
	  end
      
      elseif strcmp(tipo,'class')
          for j=1:M
            %%% Complete el codigo %%%
            sum = 0;
            for i = 1:N
                prueba = Xval(j,:)-Xent(i,:);   
                prueba = sqrt(norm(prueba));
                prueba = prueba/h;
                prueba = gaussianKernel(prueba);
                sum = sum+prueba;
            end
            sum = sum/N;
            Yesti(j)=sum;
            %%%%%%%%%%%%%%%%%%%%%%%%%%
          end
	  
      end

  end
