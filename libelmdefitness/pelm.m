%function [TrainingTime, TrainingAccuracy, TestingAccuracy] = ...
function [TrainingTime, ConfusionMatrixTrain, ConfusionMatrixTest, CCRTrain, MSTrain, CCRTest, MSTest...
    	NumberofInputNeurons,NumberofHiddenNeurons,NumberofHiddenNeuronsFinal,NumberofOutputNeurons,InputWeight,OutputWeight,pstar_train] = ...
    pelm(train_data, test_data, Elm_Type, NumberofHiddenNeurons, ActivationFunction,  wMin, wMax,EE)

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




% voy a podar

% fracciono el dataset  75 y 25%
percent = 0.75;
numpatterns = size(H,2);
numhiddenneurons = size(H,1);


%csp = floor(percent*numpatterns);
%NumpatternsComprobation = numpatterns - csp;
%indices= randperm(numpatterns);
%indicesT=indices(1:csp);
%indicesC=indices(csp+1:numpatterns);


Tone = zeros(numpatterns,1);
for i = 1 : numpatterns
   
    mm = find(T(:,i)==1);
    mm = mm(1);    
    Tone(i) = mm;
end

numclasses = size(unique(Tone),1);

cv = cvpartition(Tone, 'holdout', 1-percent);
indicesC=(find(cv.test==1));
indicesT=(find(cv.test==0));
indices=[indicesT' indicesC']';
csp=size(indicesT,1);
NumpatternsComprobation = numpatterns - csp;



HT = H(:,indicesT) ;
HC = H(:,indicesC);





TT = T(:,indicesT);
TC = T(:,indicesC);


%load iris.dat
%cv = cvpartition(iris(:, 1), 'holdout', 0.25);
%text_mat = iris(cv.test, :);

%cv.test






numbin = 15;

numattrs = size(InputWeight,2);  
featuresEval = zeros(numhiddenneurons,1);

TTone = zeros(csp,1);
for i = 1 : csp
   
    mm = find(T(:,i)==1);
    mm = mm(1);    
    TTone(i) = mm;
end

%size(TT)
%size(T)


%[TT  T']

for i = 1 : numhiddenneurons
    featuresEval(i) = chi2featureMine(discretize(HT(i,:),numbin)', TTone', numbin,numclasses) ;
end

%maxim = max(featuresEval);
%minim = min(featuresEval);
%featuresEval = (featuresEval-minim)/(maxim-minim)

featuresEval = featuresEval/sum(featuresEval); 

%inds= [1:numhiddenneurons];


temporalH = [HT featuresEval               [1:numhiddenneurons]'  HC];
temporalH = sortrows(temporalH,csp+1)';   % ordenar por la evaluacion de las caracteristicas

evals= temporalH(csp+1,:)';
nuevosindices= temporalH(csp+2,:)';

ordtrash=sort(featuresEval);
HTO = temporalH(1:csp,:);
HCO = temporalH(csp+3:end,:);
%size(HTO);


%[evals ordtrash];

 

step = 0.1;  % proposed in paper
suma = 0;
cont=0;
rm=1;
rM=9;
evals=evals/sum(evals);

%nuevosindices
max(nuevosindices);
min(nuevosindices);
InputWeight=InputWeight(nuevosindices',:);   % ordeno los pesos
H=H(nuevosindices',:);
BiasofHiddenNeurons = BiasofHiddenNeurons(nuevosindices',:);

MinAIC = 1000000;
MinAICNeuronQuantityI=-1;

for i = numhiddenneurons-1: -1 : 2 % proposed in paper
  suma = suma + evals(i);
 % suma/step
  if( floor(suma/step) ~= cont  )
      cont=floor(suma/step);
      if (cont>=rm) && (cont<=rM)
        aHT = HTO(:,i:numhiddenneurons);
        aHCO=HCO(:,i:numhiddenneurons);
        
       
        
        %calcular   en Entrenamiento 
         B =pinv(aHT) * TT';
        %aY= (aHT*B)';
        %[winnerTrain LabelTrainPredicted] = max(aY);        
        %[ff predict] = max(TT); 
        %[predict' LabelTrainPredicted' aY'];
        
        
        %calcular en test
      
        
    
      
        aYTest= (aHCO*B)';
        [winnerTrain LabelTrainPredictedTemp] = max(aYTest);        
        [ff targets] = max(TC); 
        %[predict' LabelTrainPredictedTemp' aYTest'];
        
        
        
        %T_org=targets';
        %LabelTrainPredictedTemp = LabelTrainPredictedTemp';    
        CM = confmat(targets'-1,LabelTrainPredictedTemp'-1);
    
       % if size(CM,1)==4
       %     unique(targets)
       %     fffff=3;
       % end
        
        aCCR = CCR(CM); %*100        
        S_i = numhiddenneurons-i+1;
        asigma_square=(1-aCCR)*(1-aCCR)* NumpatternsComprobation*NumpatternsComprobation; 
        AIC = 2*NumpatternsComprobation*log(asigma_square/NumpatternsComprobation)+S_i;
        
        if(AIC<MinAIC)
            MinAIC=AIC;
            MinAICNeuronQuantityI=i;
            
        end
        
      end
  end
end

%MinAICNeuronQuantityI=5;

H=H(MinAICNeuronQuantityI:numhiddenneurons,:);
InputWeight=InputWeight(MinAICNeuronQuantityI:numhiddenneurons,:);            
BiasofHiddenNeurons=BiasofHiddenNeurons(MinAICNeuronQuantityI:numhiddenneurons,:);            
NumberofHiddenNeuronsFinal= numhiddenneurons - MinAICNeuronQuantityI+1;   


clear tempH;                                        %   Release the temnormMinrary array for calculation of hidden neuron output matrix H


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
    
