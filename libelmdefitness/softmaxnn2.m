function s = softmaxnn2(X)

[NRow NCol] = size(X);

s = zeros(NRow,NCol);

Numer = exp(X);
NumerTrans = Numer';

SumProb = sum(NumerTrans);
SumProb = SumProb';

for i=1:NRow
    s(i,:) = Numer(i,:) ./ SumProb(i,1);
end;

