function W = regresionLogistica(X,Y,eta)

[N,D]=size(X);
W = zeros(D,1);
W = W';
for iter = 1:400
    %%% Completar el c�digo %%% 
    
    W = W -((eta/N)*(((sigmoide(W*X'))'-Y)'*X));
    %%% Fin de la modificaci�n %%%
end
W = W';
end