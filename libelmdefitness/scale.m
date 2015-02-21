function [XN,TN, xmin,xmax] = scale(X,T,XNmin,XNmax)

xminTrain = min(X);
xminTest = min(T);
Vmin = [xminTrain;xminTest];
xmin = min(Vmin);

xmaxTrain = max(X);
xmaxTest = max(T);
Vmax = [xmaxTrain;xmaxTest];
xmax = max(Vmax);

XN = zeros(size(X));
TN = zeros(size(T));
for j=1:size(X,2)
    XN(:,j) = ((XNmax-XNmin) / (xmax(j)-xmin(j))) * (X(:,j)-xmin(j)) + XNmin;
    TN(:,j) = ((XNmax-XNmin) / (xmax(j)-xmin(j))) * (T(:,j)-xmin(j)) + XNmin;
end

