function w=gradienteConError(X,Y)
    [N,d] = size(X);
    w = rand(d+1,1);
    maxIter = 200;
    eta  = 0.01;
    x2 = [X , ones(N,1)];
    errorVec = [];
    index = [];
    
    for iter = 1:maxIter
        
       %------------------------------
       % una figura con dos graficos
       subplot(1,2,1);
       plot(X(1:100, 1),X(1:100,2), 'xr');
       hold on;
       plot(X(101:end, 1),X(101:end,2), 'xb');
       %------------------------------
        
        
       plot(X(1:100,1),X(1:100,2),'xr');
        hold on
        plot(X(101:end,1),X(101:end,2),'xb');
        
        x = linspace(min(X(:,1)),max(X(:,1)),10);
        y1 = -(w(1)/w(2)) * x-(w(3)/w(2));
%         plot(x,y1,'k');
%         hold off;
        error=0; %Criterio de error
        for j=1:d+1
            sum=0;
            for i=1:N
                sum2=0;
                for k=1:d+1
                    sum2 = sum2+w(k)*x2(i,k);
                end
                sum=sum+misigmoide(sum2)-Y(i)*x2(i,j);
                error = error + (-Y(i)*log(misigmoide(sum2))-(1-Y(i))*log(1-misigmoide(sum2))); %Acomulando el error
            end
            w(j)=w(j)-eta*sum/N;
            error = error/N;
        end
        errorVec = [errorVec, error];
        index = [index, iter];
        subplot(1,2,2);
        plot(index, errorVec);
        pause(0.1);
    end
end

            