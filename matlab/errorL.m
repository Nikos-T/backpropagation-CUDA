function deltaL = errorL(aL, zL, y)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
deltaL = (aL-y)./(1+exp(-zL));

end

