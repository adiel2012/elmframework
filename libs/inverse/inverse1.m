function [ INV ] = inverse1( H, HI, d )
%INVERSE1 Summary of this function goes here
%   Detailed explanation goes here

%H= rand(200,10);

numpatterns = size(H,1);


%d=rand(numpatterns,1);

H2=[H d];

%HI = (inv(H'*H)*H')' ; % pinv(H');

D=((d'*((eye(numpatterns)-H*HI')))') /(d'*((eye(numpatterns)-H*HI'))*d);

U= HI'*(eye(numpatterns)-d*D') ;

INV = [U' D]' ;% inv(H2'*H2)*H2' % [U' D]

%II=pinv(H2) ;

%sum(sum( abs( INV- II) ));

end

