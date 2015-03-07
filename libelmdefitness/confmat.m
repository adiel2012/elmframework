function c = confmat(x,y)

minx = min(x);
maxx = max(x);


c = zeros(maxx-minx+1);
for i = minx:maxx
   index = x == i;
   for j = minx:maxx
      z = y(index);
      c(i-minx+1,j-minx+1) = length(find(z == j));
   end
end

g=1;

