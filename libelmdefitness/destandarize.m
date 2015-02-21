function [XN] = destandarize(X,XMeans,XStds)

XN = zeros(size(X));
for i=1:size(X,2)
    XN(:,i) = (X(:,i)* XStds(i)) + XMeans(i);
end
