function bitVal=entropyF(classVec,featVec)
%assuming that the classVec and feature are enumerated 
%each different element takes values from 1-totNumDiff Values
featTot=length(featVec);
ithEle=1;
sortClassVec=sort(classVec);
totClassTypes=sortClassVec(end);
subFeatInd=find(featVec==ithEle);
subFeatTot=length(subFeatInd);
size(featVec);
while(subFeatTot~=0)
	proportion=(subFeatTot)/featTot;
	%within the subFeature domain I need to find the to number of  classes that exist
	%so you make a histogram of this domain
	classTot=zeros(totClassTypes,1);
	for(i=1:subFeatTot)
		classKind=classVec(subFeatInd(i));
		classTot(classKind)=classTot(classKind)+1;
	end
	
	for(i=1:totClassTypes)
        if(classTot(i)~=0)
            logVal(i)=-(classTot(i)/subFeatTot)*log2((classTot(i)/subFeatTot));
        else
            logVal(i)=0;
        end
    end
	ent(ithEle)=proportion*sum(logVal);
	%update to the next element in this feature
	ithEle=ithEle +1;
	subFeatInd=find(featVec==ithEle);
	subFeatTot=length(subFeatInd);
end
bitVal=sum(ent);