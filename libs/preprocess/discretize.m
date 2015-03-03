function [ result ] = discretize( x, numbin )
%DISCRETIZE Summary of this function goes here
%   Detailed explanation goes here

minimun = min(x);
maximun = max(x);
N = size(x,2);
result = zeros(N,1);

for i = 1 : N
   result(i) = floor((x(i)-minimun)/(maximun-minimun) * numbin) ;
   result(i) = min(result(i),numbin-1);
end

end

%  discretize([0 1 2 3 4 5 6 7 8 9 9 9 9 9 9 9 9 9 9 9],3)