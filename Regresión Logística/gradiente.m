function w=gradiente(X,Y)
    %N -> Numero de datos
    %d -> Cantidad de variables
    [N,d] = size(X);
    %Valores iniciales para W
    w = rand(1,d+1);
    %Cantidad de iteraciones y saltos entre cada iteración
    maxIter = 200; eta  = 0.01;
    Error=[];
    Itera=[];}
    %Agregamos una columna de 1 ya que x para W0 es 1
    x2 = [X , ones(N,1)];
    figure(1)
    for iter = 1:maxIter %for iteraciones
        %----Grafica los datos
        subplot(1,2,1);
        plot(X(1:100,1),X(1:100,2),'xr');
        hold on
        plot(X(101:end,1),X(101:end,2),'xb');
        %----Graficar la linea de división---
        x=linspace(min(X(:,1)),max(X(:,1)),10);
        y=-(w(1)/w(2)).*x-(w(3)/w(2));
        plot(x,y,'k');
        hold off
        %----Graficar el error-----
        subplot(1,2,2)
        Itera = [Itera,iter];
        error = ErrorFunc(N,Y,x2,w);
        Error=[Error,error];
        plot(Itera,Error);
        pause(0.5);
        
        %Función de actualización del vector W
        w = w -((eta/N)*(((misigmoide(w*x2'))'-Y)'*x2));
    end
end

            