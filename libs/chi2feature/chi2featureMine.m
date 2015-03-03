function [chi_square] = chi2featureMine(feature, label, numbin)
% chi squared feature measure for multinomial class
% find class name
class = unique(label);
N = size(feature, 2);
numclases = size(class,2);

%data=[feature' label'];
sorted = sortrows([feature' label'],1);

minimun = min(feature);
maximun = max(feature);


for i = 1 : N
   sorted(i,1) = floor((sorted(i,1)-minimun)/(maximun-minimun) * numbin) ;
   sorted(i,1) = min(sorted(i,1),numbin-1);
end

A=zeros(numbin,numclases);
R=zeros(numclases,1);
Z=zeros(numbin,1);

for i = 1 : N
    A(sorted(i,1)+1,sorted(i,2)) = A(sorted(i,1)+1,sorted(i,2)) + 1;
    Z(sorted(i,1)+1) = Z(sorted(i,1)+1)   +  1;
    R(sorted(i,2)) = R(sorted(i,2)) +  1;
end
 E=zeros(size(A));
% calculate E
for i = 1 :  numclases
    for j = 1 : numbin
       E(j,i) = (R(i)*Z(j))/N;
    end
end


chi_square = 0;
for i = 1 : numbin
    for j = 1 : numclases
      if E(i,j) ~= 0
           chi_square = chi_square + (A(i,j)-E(i,j))^2/E(i,j);        
       end
    end
end

end


% http://www.mathworks.com/matlabcentral/fileexchange/28012-chi-square-feature-analysis/content/chi2feature.m


    
    
    
    
    
    
    
    
    
    
    