function [ Error ] = Criterio(Xtrain, Ytrain, Xtest , Ytest )
%CRITERIO Summary of this function goes here
%   Detailed explanation goes here
    Yest = classify( Xtest , Xtrain , Ytrain);
    Error=sum( Ytest ~= Yest )  / length ( Yest) ;
end