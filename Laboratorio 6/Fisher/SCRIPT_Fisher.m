clear;

load('DatosSeleccion.mat');

indPearson  = corrcoef([X,Y]);
dataForClases = classificatedata(X, Y);
fishers = calculatefisher(dataForClases);