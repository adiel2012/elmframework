function [MS, class, nOfBadPatterns] = Sensitivity(C)

MS = 1.0;
%class = -1.0;
nOfBadPatterns = 0;
for i=1:length(C)
    nOfPatterns = 0;
    for j=1: length(C)
        nOfPatterns = nOfPatterns + C(i,j);
    end;
    sensitivity = C(i,i) / nOfPatterns;
    
    if(sensitivity < MS)
        MS = sensitivity;
        %class = i;
        %nOfBadPatterns = nOfPatterns;
    end;
    
end;

