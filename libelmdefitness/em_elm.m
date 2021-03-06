%function [TrainingTime, TrainingAccuracy, TestingAccuracy] = ...
function [TrainingTime, ConfusionMatrixTrain, ConfusionMatrixTest, CCRTrain, MSTrain, CCRTest, MSTest...
    NumberofInputNeurons,NumberofHiddenNeurons,NumberofHiddenNeuronsFinal,NumberofOutputNeurons,InputWeight,OutputWeight,pstar_train] = ...
    em_elm(train_data, test_data, Elm_Type, NumberofHiddenNeurons, ActivationFunction,  wMin, wMax,EE)
%function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction, normMin,normMax, wMin, wMax)

% Usage: elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
%
% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%                           'tribas' for Triangular basis function
%                           'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
%
% Output:
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy:
%                           RMSE for regression or correct classification rate for classification
% TestingAccuracy       - Testing accuracy:
%                           RMSE for regression or correct classification rate for classification
%
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class
%
% Sample1 regression: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm('sinc_train', 'sinc_test', 0, 20, 'sig')
% Sample2 classification: elm('diabetes_train', 'diabetes_test', 1, 20, 'sig')
%
%%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
%%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
%%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
%%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
%%%%    DATE:       APRIL 2004

%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;

%%%%%%%%%%% Load training dataset
%train_data=load(TrainingData_File);
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
%test_data=load(TestingData_File);
TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                    %   Release raw testing data array

%%%%%%%%%%% Filter unmeaninful attributes
offset = 0;
attrs = size(P,1);

while offset  < attrs
    offset = offset + 1;
    if (max(P(offset,:)) == min(P(offset,:)))
        % Delete attribute both in train and test sets
        P(offset,:) = [];
        TV.P(offset,:) = [];
        offset = offset-1;
        attrs = attrs-1;
    end
end

%%%%%%%%%%% Check for constant attributes that we can delete. Otherwise a
%%%%%%%%%%% NaN can be obtained later.
% minvals = min(P');
% maxvals = max(P');
%
% r = 0;
% for k=1:size(P,1)
%     if minvals(k) == maxvals(k)
%         r = r + 1;
%         index(r) = k;
%     end
% end
%
% if r > 0
% 	r = 0;
% 	for k=1:size(index,2)
% 	    P(index(k)-r,:) = [];
% 	    TV.P(index(k)-r,:) = [];
% 	    r = r + 1;
% 	end
% end
% clear index;clear r;clear minvals;clear minvals;

%------Perform log(P) calculation once for UP
% if strcmp(ActivationFunction, 'up')
%    P = log(P);
%    TV.P = log(TV.P);
% end

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    
    %%%%%%%%%% Processing the targets of training
    nOfPatterns = zeros(number_class,1);
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                nOfPatterns(j,1) = nOfPatterns(j,1) + 1;
                break;
            end
        end
        temp_T(j,i)=1;
    end
    T_org = T';
    T=temp_T*2-1;
    
    pstar_train = min(nOfPatterns) / NumberofTrainingData;
    
    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break;
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV_org = TV.T';
    TV.T=temp_TV_T*2-1;
    
end                                                 %   end if of Elm_Type

%%%%%%%%%%% Calculate weights & biases
%start_time_train=cputime;
tStart = tic;

%------Perform log(P) calculation once for UP
% The calculation is done here for including it into the validation time
if strcmp(ActivationFunction, 'up')
    P = log(P);
    TV.P = log(TV.P);
end

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons

switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
        
        BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
        tempH=InputWeight*P;
        %Movido abajo
        %clear P;                                            %   Release input of training data
        ind=ones(1,NumberofTrainingData);
        BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
        tempH=tempH+BiasMatrix;
    case {'up'}
        InputWeight = wMin + (wMax-wMin).*rand(NumberofHiddenNeurons,NumberofInputNeurons);
    case {'rbf'}
        P = P';
        if (NumberofTrainingData>2000)
            TY=pdist(P(randperm(2000),:));
        else
            TY=pdist(P);
        end
        a10=prctile(TY,20);
        a90=prctile(TY,60);
        MP=randperm(NumberofTrainingData);
        W1=P(MP(1:NumberofHiddenNeurons),:);
        W10=rand(1,NumberofHiddenNeurons)*(a90-a10)+a10;
        InputWeight = [W1 W10'];
    case {'rbf2'}
        P = P';
        
        % REDISTRIBUIR CLASES
        CLASS0 = P(T_org==0,:);
        CLASS1 = P(T_org==1,:);
        CLASS2 = P(T_org==2,:);
        
        if (NumberofTrainingData>2000)
            TY=pdist(P(randperm(2000),:));
        else
            TY=pdist(P);
        end
        a10=prctile(TY,20);
        a90=prctile(TY,60);
        
        %         Class0Weight = 0.5;
        %
        %         Class0Number = round(NumberofHiddenNeurons * Class0Weight);
        %         Class1Number = round(NumberofHiddenNeurons * (1-Class0Weight));
        
        Class0Number = round(NumberofHiddenNeurons / 3);
        Class1Number = round(NumberofHiddenNeurons / 3);
        Class2Number = round(NumberofHiddenNeurons / 3);
        
        MP=randperm(size(CLASS0,1));
        W1a=CLASS0(MP(:,1:Class0Number),:);
        
        MP=randperm(size(CLASS1,1));
        W1b=CLASS1(MP(:,1:Class1Number),:);
        
        MP=randperm(size(CLASS2,1));
        W1c=CLASS2(MP(:,1:Class2Number),:);
        
        %W1 = [W1a' W1b']';
        W1 = [W1a' W1b' W1c']';
        
        W10=rand(1,NumberofHiddenNeurons)*(a90-a10)+a10;
        InputWeight = [W1 W10'];
    case {'krbf'}
        P = P';
        opts = statset('MaxIter',200);
        [IDX, C, SUMD, D] = kmeans(P,NumberofHiddenNeurons,'Options',opts);
        MC = squareform(pdist(C));
        MCS = sort(MC);
        MCS(1,:)=[];
        radii = sqrt(MCS(1,:).*MCS(2,:));
        InputWeight = [C radii'];
        
        W1 = C;
        W10 = radii;
    case {'grbf'}
        MP = randperm(NumberofTrainingData);
        InputWeight = P(:,MP(1:NumberofHiddenNeurons))';
        
end




%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here
    case {'up'}
        %PU_j(X) = productorio_{i=0}^n (x_i^{w_{ji}})
        %P = log(P);
        
        H = zeros(NumberofHiddenNeurons,NumberofTrainingData);
        for i = 1 : NumberofTrainingData
            for j = 1 : NumberofHiddenNeurons
                temp = zeros(NumberofInputNeurons,1);
                for n = 1: NumberofInputNeurons
                    temp(n) = InputWeight(j,n)*P(n,i);
                end
                H(j,i) =  sum(temp);
            end
        end
        clear temp;
    case {'rbf','krbf','rbf2'}
        % TODO: Un hack
        H = zeros(NumberofTrainingData,NumberofHiddenNeurons);
        for j=1:NumberofHiddenNeurons
            H(:,j)=gaussian_func(P,W1(j,:),W10(1,j));
            %KM.valueinit(:,j)=gaussian_func(x,W1(j,:),W10(1,j));
        end
        H = H';
    case {'grbf'}
        % Compute Pairwise Euclidean distance
        EuclideanDistanceArray = pdist(InputWeight);
        EuclideanDistanceMatrix = squareform(EuclideanDistanceArray);
        EuclideanDistanceSorted = sort(EuclideanDistanceMatrix);
        % Larges distances and nearest distances
        dF = EuclideanDistanceSorted(2,:);
        %dN = (dF*0.05)/0.95;
        dN = ones(size(dF)) * sqrt((0.001^2) * NumberofInputNeurons);
        % Determine Tau and radii values
        %taus = 4.0674 ./ (log(dF./dN));
        taus = 5.6973 ./ (log(dF./dN));
        
        taus = ones(1,NumberofHiddenNeurons)*2;
        %radii = dF ./(-log(0.95)).^(1 ./taus);
        radii = dF ./(-log(0.99)).^(1 ./taus);
        % Obtain denominator
        denominator = radii .^taus;
        denominator_extended = repmat(denominator,NumberofTrainingData,1)';
        % Obtain Numerator
        EuclideanDistance = pdist2(InputWeight,P','euclidean');
        taus_extended = repmat(taus,NumberofTrainingData,1)';
        numerator = EuclideanDistance.^taus_extended;
        % Calculate Hidden Node outputs
        H = exp(-(numerator./denominator_extended));
end
%COMENTADO clear P;

clear tempH;                                        %   Release the temnormMinrary array for calculation of hidden neuron output matrix H





numpatterns = size(H,2);
numhiddenneurons = size(H,1);

y = zeros(numpatterns,1);
for i = 1 : numpatterns
    
    mm = find(T(:,i)==1);
    mm = mm(1);
    y(i) = mm;
end


%y = species;
numfolds = 2;
c = cvpartition(y,'k',numfolds);
bestthresold = -1;
maxsumaCCR=-1;
ini=3;
numoutputneurons=size(T,1);

%Calculo error para 100 neuronas y para 3 neuronas
aH=H;
aBtemp= pinv(aH')*T';

% aH*aBtemp
% aH'*aBtemp
% aH'*aBtemp'
% aH*aBtemp'

error100 = sum(sum(abs( aH'*aBtemp -T' ))) / (numpatterns*numoutputneurons);

aH=H(1:ini,:);
aBtemp= pinv(aH')*T';
error3 = sum(sum(abs( aH'*aBtemp -T' ))) / (numpatterns*numoutputneurons);
%numthresolds = 5;

%arrayerrorthresold = zeros(numthresolds,1);
%for i = 1: numthresolds
%    arrayerrorthresold(i)= error3 - (i)*(error3-error100)/(numthresolds);
%end
%arrayerrorthresold

%arrayerrorthresold = [0.2 0.15 0.1]; %[0.5 0.4 0.3 0.2 0.15 0.12 0.1 0.08 0.06];
%cantthresolds = size(arrayerrorthresold,2);
%numpatterns= size(H,2);
%numoutputneurons=size(T,1);
errors = ones(numfolds,NumberofHiddenNeurons)*1000000;
ccrs = ones(numfolds,NumberofHiddenNeurons)*1000000;

for indexfold = 1 : numfolds
    
    indicesC=(find(c.test(indexfold)==1));
    indicesT=(find(c.test(indexfold)==0));
    
    HSub=H(:,indicesT);
    HSubC=H(:,indicesC);
    TSub=T(:,indicesT);
    CSub=T(:,indicesC);
    
    numpatterns= size(HSub,2);
    numoutputneurons=size(TSub,1);
%    error = errorthresold+1; % para que entre la primera ves
    i=3;
    %BtempAnterior=0;
    Btemp=-1;
    ini=3;
    for i=ini : NumberofHiddenNeurons
        HTemporal = HSub(1:i-1,:)';
        HTemporalC = HSubC(1:i-1,:);
        if(i==ini)
            HI=inv(HTemporal'*HTemporal)* HTemporal';
        else
            d =  HSub(i-1,:)';
            D=((d'*((eye(numpatterns)-HTemporalAnterior'*HI)))') /(d'*((eye(numpatterns)-HTemporalAnterior'*HI))*d);
            U= HI*(eye(numpatterns)-d*D') ;
            HI = [U' D]' ;
            %HI2 =  inverse1( HTemporalAnterior', HI', d);
            %numpatterns
            %numoutputneurons
            %error=sum(sum(abs(HI-pinv( [HTemporalAnterior' d]  ))))/(size(HI,1)*size(HI,2))
        end
        
        
        Btemp= HI*TSub';
        error = sum(sum(abs( HTemporal*Btemp -TSub' ))) / (numpatterns*numoutputneurons);
        errors(indexfold,i)= error;
        
        NumberofHiddenNeuronsFinal=i;
        HTemporalAnterior =HTemporal';
        
        
         aYTest= (HTemporalC'*Btemp)';
        [winnerTrain LabelTrainPredictedTemp] = max(aYTest);
        [ff targets] = max(CSub);
        CM = confmat(targets'-1,LabelTrainPredictedTemp'-1);
        aCCR = CCR(CM);
        ccrs(indexfold,i) = aCCR;
    end
    
    % calcular CCR  sobre el conjunto de comprobación
    aB =  Btemp;
    
    % if(i <= NumberofHiddenNeurons && i~=ini+1)
    %     aB= BtempAnterior;
    % else
    %     aB=  Btemp;
    % end
    
    
    
end




positions= ones(numfolds,1)*ini;  % neurona por la que va
%currenterror = max(errors(:,ini));
maxCCR= -1    ;%sum(ccrs(:,ini));
bestthresold =-1;


g=0;


for cont = 1 : (NumberofHiddenNeurons-ini)*numfolds
    
    % mover las posiciones
         %se calcula el menor de los siguientes errores de cada fold
         pos=-1;
         mineerror=1000000000000;
         for j = 1 : numfolds
             if(positions(j)< NumberofHiddenNeurons)
                 %ccrs(j,positions(j)+1)
                 if(mineerror>errors(j,positions(j)+1))
                     mineerror=errors(j,positions(j)+1);
                     pos=j;
                 end
             end
         end
         %(NumberofHiddenNeurons-ini+1)*numfolds
         %g=g+1
         %if(pos==-1)
         %    gfhf;
         %end
         %pos;
         positions(pos)=positions(pos)+1;
    %calcular el ccr
    ccrsum=0;
    for j = 1 : numfolds
        ccrsum = ccrsum+ccrs(j,positions(j));
    end
    if(maxCCR<ccrsum)
        maxCCR=ccrsum;
        bestthresold= mineerror;
        %positions
    end
end


%ccrs(:,ini)
%ccrs(:,100)

%termine de estimar el thresold
errorthresold=bestthresold;




ini=3;
numpatterns= size(H,2);
numoutputneurons=size(T,1);
error = errorthresold+1;
i= ini;
while i <= NumberofHiddenNeurons && error > errorthresold
    HTemporal = H(1:i-1,:)';
    if(i==ini)
        HI=inv(HTemporal'*HTemporal)* HTemporal';
    else
        d =  H(i-1,:)';
        D=((d'*((eye(numpatterns)-HTemporalAnterior'*HI)))') /(d'*((eye(numpatterns)-HTemporalAnterior'*HI))*d);
        U= HI*(eye(numpatterns)-d*D') ;
        HI = [U' D]' ;
    end
    
    
    Btemp= HI*T';
    error = sum(sum(abs( HTemporal*Btemp -T' ))) / (numpatterns*numoutputneurons);
    
    NumberofHiddenNeuronsFinal=i;
    i = i+1;
    HTemporalAnterior =HTemporal';
end

























H=H(1:min(i,NumberofHiddenNeuronsFinal),:);
BiasofHiddenNeurons=BiasofHiddenNeurons(1:min(i,NumberofHiddenNeuronsFinal));
InputWeight=InputWeight(1:NumberofHiddenNeuronsFinal,:);








%0.1235    0.3919
%0.3111    0.5226
%0.4838    0.7770
%0.1299    0.4014
%0.4204    0.9349

%NumberofHiddenNeurons =NumberofHiddenNeuronsFinal;

%size(BiasofHiddenNeurons)


%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
OutputWeight=pinv(H') * T';                        % slower implementation
% OutputWeight=inv(H * H') * H * T';                         % faster implementation
%end_time_train=cputime;
%TrainingTime=end_time_train-start_time_train;        %   Calculate CPU
%time (seconds) spent for training ELM

TrainingTime = toc(tStart);

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
if Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y));               %   Calculate training accuracy (RMSE) for regression case
end
clear H;

%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;

if strcmpi(ActivationFunction, 'sig')
    tempH_test=InputWeight*TV.P;
    %Movido abajo
    %clear TV.P;             %   Release input of testing data
    ind=ones(1,NumberofTestingData);
    
    BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
    
    if( size(tempH_test,1)~= size(BiasMatrix,1) || size(tempH_test,2)~= size(BiasMatrix,2) )
        dfgdfg=7;
    end
    
    tempH_test=tempH_test + BiasMatrix;
end

switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);
        %%%%%%%% More activation functions can be added here
    case {'up'}
        
        %TV.P = log(TV.P);
        H_test = zeros(NumberofHiddenNeurons, NumberofTestingData);
        
        for i = 1 : NumberofTestingData
            for j = 1 : NumberofHiddenNeurons
                temp = zeros(NumberofInputNeurons,1);
                for n = 1: NumberofInputNeurons
                    %temp(n) = TV.P(n,i)^InputWeight(j,n);
                    temp(n) = InputWeight(j,n)*TV.P(n,i);
                end
                %H_test(j,i) =  prod(temp);
                H_test(j,i) =  sum(temp);
            end
        end
        
        clear temp;
    case {'rbf','krbf','rbf2'}
        H_test = zeros(NumberofTestingData,NumberofHiddenNeurons);
        TV.P = TV.P';
        
        for j=1:NumberofHiddenNeurons
            H_test(:,j)=gaussian_func(TV.P,W1(j,:),W10(1,j));
        end
        H_test = H_test';
        
    case {'grbf'}
        % Repmat denominator to Testing data
        denominator_extended = repmat(denominator,NumberofTestingData,1)';
        % Recalculate Euclidean Distance
        EuclideanDistanceTest = pdist2(InputWeight,TV.P','euclidean');
        taus_extended = repmat(taus,NumberofTestingData,1)';
        numerator = EuclideanDistanceTest.^taus_extended;
        % Calculate Hidden Node outputs
        H_test = exp(-(numerator./denominator_extended));
end

clear TV.P;             %   Release input of testing data


TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data

end_time_test=cputime;
TestingTime=end_time_test-start_time_test;           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

clear H_test;

if Elm_Type == REGRESSION
    TestingAccuracy=sqrt(mse(TV.T - TY));            %   Calculate testing accuracy (RMSE) for regression case
end

if Elm_Type == CLASSIFIER
    
    [winnerTrain LabelTrainPredicted] = max(Y);
    [winnerTest LabelTestPredicted] = max(TY);
    
    LabelTrainPredicted = LabelTrainPredicted';
    LabelTrainPredicted = LabelTrainPredicted -1;
    LabelTestPredicted = LabelTestPredicted';
    LabelTestPredicted = LabelTestPredicted -1;
    
    ConfusionMatrixTrain = confmat(T_org,LabelTrainPredicted);
    ConfusionMatrixTest = confmat(TV_org,LabelTestPredicted);
    
    CCRTrain = CCR(ConfusionMatrixTrain)*100;
    CCRTest = CCR(ConfusionMatrixTest)*100;
    
    MSTrain = Sensitivity(ConfusionMatrixTrain)*100;
    MSTest = Sensitivity(ConfusionMatrixTest)*100;
    
    %%%%%%%%%% Calculate training & testing classification accuracy
    %     MissClassificationRate_Training=0;
    %     MissClassificationRate_Testing=0;
    %
    %     for i = 1 : size(T, 2)
    %         [x, label_index_expected]=max(T(:,i));
    %         [x, label_index_actual]=max(Y(:,i));
    %         if label_index_actual~=label_index_expected
    %             MissClassificationRate_Training=MissClassificationRate_Training+1;
    %         end
    %     end
    %     TrainingAccuracy=1-MissClassificationRate_Training/size(T,2);
    %     for i = 1 : size(TV.T, 2)
    %         [x, label_index_expected]=max(TV.T(:,i));
    %         [x, label_index_actual]=max(TY(:,i));
    %         if label_index_actual~=label_index_expected
    %             MissClassificationRate_Testing=MissClassificationRate_Testing+1;
    %         end
    %     end
    %     TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);
end

