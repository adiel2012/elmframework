function [BIG ind]=getBestEnt(classVec,allFeat)
	[maxR totNumFeat]=size(allFeat);
	for(i=1:totNumFeat)
        n=normFeat(allFeat(:,i));
		IG(i)=entropy(classVec)-entropyF(classVec,n);
	end
	[BIG ind]=max(IG);
