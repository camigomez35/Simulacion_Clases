%Medir el error 
function error = ErrorFunc(N,Y,x,w)
    %Y y X datos
    %Vecator W
    %N numero de datos
    a = -Y'*log(misigmoide(w*x'))';
    b = (1-Y)'*(log(1-misigmoide(w*x')))';
    error = a - b;
    error = 1/N * error;
end

