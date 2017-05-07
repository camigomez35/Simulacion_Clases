function W = regresionMultiple(X,Y,eta)

[N,D]=size(X);
W=zeros(D,1);
W = W';

for iter = 1:200
    %%% Completar el cï¿½digo %%% 
    
    %Función de actualización del vector W
    W = W -((eta/N)*(((W*X'))'-Y)'*X);
    
    %%% Fin de la modificaciï¿½n %%%
end
% figure
% i=1:length(W);
% plot(i,W(i),'xr');
W = W';
end