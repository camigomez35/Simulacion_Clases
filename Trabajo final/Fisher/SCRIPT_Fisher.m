clear;

load('HOGFeatures.mat');
X = caracteristicas;
load('clases.mat');
Y = clases;

[XNormal,mu,sigma] = zscore(X);

indPearson  = corrcoef([XNormal,Y]);
dataForClases = classificatedata(XNormal, Y);
fishers = calculatefisher(dataForClases);