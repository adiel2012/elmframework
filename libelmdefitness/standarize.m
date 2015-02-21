function [XN, XMeans, XStds] = standarize(X,XMeans,XStds)

if (nargin<3) 
    XStds = std(X);
end
if (nargin<2) 
    XMeans = mean(X);
end
XN = zeros(size(X));
for i=1:size(X,2)
    XN(:,i) = (X(:,i) - XMeans(i)) / XStds(i);
end
