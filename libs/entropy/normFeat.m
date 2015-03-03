function norm=normFeat(featVec)
%hint height([2 6 7])=[1 1 1];
unq=unique(featVec);
totUnq=length(unq);
for(i=1:totUnq)
	inds=find(featVec==unq(i));
	norm(inds)=i;
end


	
