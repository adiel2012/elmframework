function result=entropy(classVec)
	totLen=length(classVec);
	ithEle=1;
	subLen=length(find(classVec==ithEle));
	while(subLen~=0)
		logVal(ithEle)=-(subLen/totLen)*log2(subLen/totLen);
		ithEle=ithEle+1;
		subLen=length(find(classVec==ithEle));
	end
result=sum(logVal);
% reference
% http://www.mathworks.com/matlabcentral/fileexchange/14996-entropy

%Here's an example:
%hair=[1 1 2 3 2 2 2 1];
%entropyF(class,hair)
%ans =
%    0.5000
%eyes=[1 1 1 1 2 1 2 2];
%entropyF(class,eyes)
%ans =
%    0.6068
%height=[1 2 1 1 1 2 2 1];
%entropyF(class,height)
%ans =
%    0.9512
%entropy(class)
%ans =
%    0.9544
%allFeat=[eyes hair height];
%[big ind]=getBestEnt(class, allFeat)
%big =
%    0.9544
%ind

%     1
%note: big stands for best information gain
%The ind determines the 1nd feature(eyes) as the best feature











%Recommendatations:
%1. Provide a test file in the zip file.
%Your comment example is missing the variable class. I used
%class = [1 1 1 2 2 2 2 2]
%A reference to a paper, book or website which explains the algorithms would have been helpful.
%3. I get the same results (ind equal to 1) in your example if I use allFeat=[hair eyes height];
%instead of allFeat=[eyes hair height]

%I assume that the getBestEnt is not working. 

