%function [TrainingTime, TrainingAccuracy, TestingAccuracy] = ...
function [TrainingTime, ConfusionMatrixTrain, ConfusionMatrixTest, CCRTrain, MSTrain, CCRTest, MSTest...
    	NumberofInputNeurons,NumberofHiddenNeurons,NumberofHiddenNeuronsFinal,NumberofOutputNeurons,InputWeight,OutputWeight,pstar_train] = ...
    ielm(train_data, test_data, Elm_Type, NumberofHiddenNeurons, ActivationFunction,  wMin, wMax)

% esta variable la utilizan y la asigne asi
MaxNumberofHiddenNeurons = NumberofHiddenNeurons;




AresidualErrorsTotal = zeros(NumberofHiddenNeurons,1);





%-------------------------------------------------------------------------------------------------

%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;

%%%%%%%%%%% Load training dataset
%train_data=load(TrainingData_File);
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;  

%[ cantidaclases cantidadpatrones] = size(T);


%   Release raw training data array

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

clear temp_T;

L=0;    % L: number of neurons;

TrainingResidualError=ones(size(T));




%%%%%%%%%%% Calculate weights & biases
%start_time_train=cputime;
tStart = tic;

%------Perform log(P) calculation once for UP
if strcmp(ActivationFunction, 'up')
    P = log(P);
    TV.P = log(TV.P);
end

%%%%%%%%%%%%%%%%%%%%%%%%% initializing %%%%%%%%
InputWeight=zeros(NumberofInputNeurons,MaxNumberofHiddenNeurons);
HiddenBias=zeros(1,MaxNumberofHiddenNeurons);
HiddenNeuronActivation=zeros(NumberofTrainingData,MaxNumberofHiddenNeurons);
Beta=zeros(NumberofOutputNeurons,MaxNumberofHiddenNeurons);
total_cputimesofar=zeros(1,MaxNumberofHiddenNeurons);
HiddenNeuronActivationTesting=zeros(NumberofTestingData,MaxNumberofHiddenNeurons);
TrainingResidualError_Norm=zeros(1,MaxNumberofHiddenNeurons);
TestingResidualError_Norm=zeros(1,MaxNumberofHiddenNeurons);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%% Calculate the hidden node output (training)

MAXRESIDUALERROR = 0.6;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TRAINING
LASTERROR = 100000000000;
UMBRALERROR = 0.7;
%   ESTE ES EL CODIGO INSERTADO
while L<MaxNumberofHiddenNeurons && LASTERROR<UMBRALERROR
    
starting_cpu=cputime;

if L==0                        % do the thing when hidden neuron is zero.

    TrainingResidualError= T;
    L=L+1;
    switch lower(ActivationFunction)
        case {'rbf'}
            InputWeight(:,L)=2*rand(NumberofInputNeurons,1)-1; % randomly chose InputWeight for Neuron L;  for other activation functions except RBF
            HiddenBias(L)=rand(1,1);
        case {'rbf_gamma'}
            InputWeight(:,L)=2*rand(NumberofInputNeurons,1)-1; % randomly chose InputWeight for Neuron L;  for other activation functions except RBF
            HiddenBias(L)=0.5*rand(1,1);
        otherwise
            InputWeight(:,L)=2*rand(NumberofInputNeurons,1)-1; % randomly chose InputWeight for Neuron L;  for other activation functions except RBF
            HiddenBias(L)=2*rand(1,1)-1;
    end
    

    %HiddenNeuronActivation(:,L)=hidden_output(I,InputWeight(:,L),HiddenBias(L),ActivationFunction,NumberofInputNeurons)';
    HiddenNeuronActivation(:,L)=hidden_output(P,InputWeight(:,L),HiddenBias(L),ActivationFunction,NumberofInputNeurons)';

    for i=1:NumberofOutputNeurons
        SumofResidualHiddenActivation(i,:)=TrainingResidualError(i,:)*HiddenNeuronActivation(:,L);
        SumofSquareHiddenActivation=HiddenNeuronActivation(:,L)'*HiddenNeuronActivation(:,L);    
        Beta(i,L)=SumofResidualHiddenActivation(i,:)/SumofSquareHiddenActivation;
    end       

    TrainingResidualError_Previous=TrainingResidualError;   % Record (L-1)th residual error
    TrainingResidualError=TrainingResidualError_Previous-Beta(:,L)*HiddenNeuronActivation(:,L)';   % Calculate L-th residual error
    if Elm_Type==REGRESSION
        TrainingResidualError_Norm(L)=norm(TrainingResidualError)/sqrt(NumberofTrainingData);
    end

    total_cputimesofar(L)=double(cputime-starting_cpu); % CPU time spent for training
    
    for n=1:NumberofTestingData   % Calculate testing residual error after k-th neuron added. 
        TestingResidualError(:,n)= TV.T(:,n);
    end
    %HiddenNeuronActivationTesting(:,L)=hidden_output(x_testing,InputWeight(:,L),HiddenBias(L),ActivationFunction,NumberofInputNeurons)';
    HiddenNeuronActivationTesting(:,L)=hidden_output(TV.P,InputWeight(:,L),HiddenBias(L),ActivationFunction,NumberofInputNeurons)';
    
    TestingResidualError_Previous=TestingResidualError;   % Record (L-1)th residual error
    TestingResidualError=TestingResidualError_Previous-Beta(:,L)*HiddenNeuronActivationTesting(:,L)';   % Calculate L-th testing residual error
    if Elm_Type==REGRESSION
        TestingResidualError_Norm(L)=norm(TestingResidualError)/sqrt(NumberofTestingData);
    end
   
else                  % do the work when L~=0
    L=L+1;
    switch lower(ActivationFunction)
        case {'rbf'}
            InputWeight(:,L)=2*rand(NumberofInputNeurons,1)-1; % randomly chose InputWeight for Neuron L;  for other activation functions except RBF
            HiddenBias(L)=rand(1,1);
        case {'rbf_gamma'}
            InputWeight(:,L)=2*rand(NumberofInputNeurons,1)-1; % randomly chose InputWeight for Neuron L;  for other activation functions except RBF
            HiddenBias(L)=0.5*rand(1,1);
        otherwise
            InputWeight(:,L)=2*rand(NumberofInputNeurons,1)-1; % randomly chose InputWeight for Neuron L;  for other activation functions except RBF
            HiddenBias(L)=2*rand(1,1)-1;
    end

    HiddenNeuronActivation(:,L)=hidden_output(P,InputWeight(:,L),HiddenBias(L),ActivationFunction,NumberofInputNeurons)';

    for i=1:NumberofOutputNeurons
        SumofResidualHiddenActivation(i,:)=TrainingResidualError(i,:)*HiddenNeuronActivation(:,L);
        SumofSquareHiddenActivation=HiddenNeuronActivation(:,L)'*HiddenNeuronActivation(:,L);    
        Beta(i,L)=SumofResidualHiddenActivation(i,:)/SumofSquareHiddenActivation;
    end      

    TrainingResidualError_Previous=TrainingResidualError;   % Record (L-1)th residual error
    TrainingResidualError=TrainingResidualError_Previous-Beta(:,L)*HiddenNeuronActivation(:,L)';   % Calculate L-th residual error
    if Elm_Type==REGRESSION
        TrainingResidualError_Norm(L)=norm(TrainingResidualError)/sqrt(NumberofTrainingData);    
    end

    total_cputimesofar(L)=total_cputimesofar(L-1)+cputime-starting_cpu; % CPU time spent for training
    
    HiddenNeuronActivationTesting(:,L)=hidden_output(TV.P,InputWeight(:,L),HiddenBias(L),ActivationFunction,NumberofInputNeurons)';
    %  HiddenNeuronActivationTesting(:,L)=hidden_output(x_testing,InputWeight(:,L),HiddenBias(L),ActivationFunction,NumberofInputNeurons)';
    TestingResidualError_Previous=TestingResidualError;   % Record (L-1)th residual error
    TestingResidualError=TestingResidualError_Previous-Beta(:,L)*HiddenNeuronActivationTesting(:,L)';   % Calculate L-th testing residual error
    if Elm_Type==REGRESSION
        TestingResidualError_Norm(L)=norm(TestingResidualError)/sqrt(NumberofTestingData);
    end
end        


   AresidualErrorsTotal(L)= sqrt( sum(sum(abs(TrainingResidualError.*abs(TrainingResidualError))))/(NumberofTrainingData*NumberofOutputNeurons));
   LASTERROR = AresidualErrorsTotal(L);
end % End while when TrainingResidualError not larger than min_goal



TrainingTime=total_cputimesofar;

if Elm_Type == REGRESSION
TrainingAccuracy=TrainingResidualError_Norm;
TestingAccuracy=TestingResidualError_Norm;
end

if Elm_Type == CLASSIFIER
    for k= 1: MaxNumberofHiddenNeurons
%         for i=1:k
%             Out_hidden_train(i,:)=hidden_output(I,InputWeight(:,i),HiddenBias(i),ActivationFunction);
%         end
        Out_train=Beta(:,1:k)*HiddenNeuronActivation(:,1:k)';
        
        MissClassificationRate_Training=0;
        MissClassificationRate_Testing=0;
        for i = 1 : size(T, 2)
            [x, label_index_expected]=max(T(:,i));
            [x, label_index_actual]=max(Out_train(:,i));
            if label_index_actual~=label_index_expected
                MissClassificationRate_Training=MissClassificationRate_Training+1;
            end
        end
        TrainingAccuracy(k) = 1-MissClassificationRate_Training/size(T,2);
    
        
    %%%%%%%%%%%%%%%%%%%%    test    %%%%%%%%%%%%
        Out_test=Beta(:,1:k)*HiddenNeuronActivationTesting(:,1:k)';
 
        for i = 1 : size(TV.T, 2)
            [x, label_index_expected]=max(TV.T(:,i));
            [x, label_index_actual]=max(Out_test(:,i));
            if label_index_actual~=label_index_expected
                MissClassificationRate_Testing=MissClassificationRate_Testing+1;
            end
        end
        TestingAccuracy(k) = 1-MissClassificationRate_Testing/size(TV.T,2);
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TRAININGEND
 %MiBeta=Beta(:,1:k);
 %           MiHiddenNeuronActivation=HiddenNeuronActivation(:,1:k);
 %           MiInputWeight=InputWeight(:,k);
 %           MiHiddenBias=HiddenBias(i);

%TestingAccuracy
%size(TrainingAccuracy)
%size(TrainingResidualError)
%size(TrainingResidualError_Norm)
%[TrainingAccuracy' TrainingResidualError_Norm' AresidualErrorsTotal]




besttraining = max(TrainingAccuracy);    % OBTENGO EL MEJOR VALOR  DE TrainingAccuracy
bests = find(TrainingAccuracy==besttraining);    % OBTENGO CUAL ES EL INDICE MEJOR   (CANTIDAD DE NEURONAS EN CAPA OCULTA = INDICE)
best_quantity_hidden_neurons=bests(1);
%best_quantity_hidden_neurons = L;


NumberofHiddenNeuronsFinal=best_quantity_hidden_neurons;  % ESTABLEZCO EL NUMERO DE NEURONAS FINAL
H = HiddenNeuronActivation(:,1:best_quantity_hidden_neurons)';

InputWeight=InputWeight(:,1:best_quantity_hidden_neurons)';
HiddenBias=HiddenBias(1:best_quantity_hidden_neurons)';

%sum(sum(HiddenBias-MiHiddenBias));

%hidden_output(P,InputWeight(:,L),HiddenBias(L),ActivationFunction,NumberofInputNeurons)';
        
%%%%%%%%%%% Calculate hidden neuron output matrix H
%switch lower(ActivationFunction)
%    case {'sig','sigmoid'}
%        %%%%%%%% Sigmoid 
%        H = 1 ./ (1 + exp(-tempH));
%        %%%%%%%% More activation functions can be added here
%    case {'up'}
%        %%%%%%%% Product Unit
%        H = exp(tempH);
%        %%%%%%%%
%     
%end
%COMENTADO clear P;

clear tempH;                                        %   Release the temnormMinrary array for calculation of hidden neuron output matrix H


%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
%OutputWeight=pinv(H') * T';                        % slower implementation
OutputWeight= Beta(:,1:best_quantity_hidden_neurons)';
TrainingTime = toc(tStart);

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
if Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y));               %   Calculate training accuracy (RMSE) for regression case
end
clear H;

%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;

%%%%%%%%%%% Calculate the Hidden node output 
%tempH_test=InputWeight*TV.P;

%switch lower(ActivationFunction)
%    case {'sig','sigmoid'}
%        %%%%%%%% Sigmoid 
%       H_test = 1 ./ (1 + exp(-tempH_test));
%    case {'up'}
%        %%%%%%%% Product Unit
%        H_test = exp(-tempH_test);
%end



H_test = hidden_output(TV.P,InputWeight',HiddenBias,ActivationFunction,NumberofInputNeurons);

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
end
    
function y1=hidden_output(x,w,b,ActivationFunction,NumberofInputs)

switch lower(ActivationFunction)
    case {'sin','sine'}
        %%%%%%%% Sines
        y1=sin(w'*x+b);     
    case {'rbf'}
        %%%%%%%% RBF
        NumberofTraining=size(x,2);
        ind=ones(1,NumberofTraining);
                            
        extend_weight=w(:,ind);%%   w is column vector
        if NumberofInputs==1
            tempH=-((x-extend_weight).^2);
        else
            tempH=-sum((x-extend_weight).^2);
        end
        

        BiasMatrix=b(:,ind);  
        tempH=tempH./BiasMatrix;
        clear extend_weight;    
        
        y1=exp(tempH)+0.0001;
    case {'rbf_gamma'}
        %%%%%%%% RBF
        NumberofTraining=size(x,2);
        ind=ones(1,NumberofTraining);
                            
        extend_weight=w(:,ind);%%   w is column vector
        if NumberofInputs==1
            tempH=-((x-extend_weight).^2);
        else
            tempH=-sum((x-extend_weight).^2);
        end

        BiasMatrix=b(:,ind);  
        tempH=tempH.*BiasMatrix;
        clear extend_weight;    
        
        y1=exp(tempH)+0.0001;         
    case {'tri'}
        %%%%%%%% Triangle
        x1=w'*x+b;
        if abs(x1)>1
            y1=0;
        elseif x1>0
            y1=1-x1;
        else
            y1=x1+1;
        end
    case {'hardlim'}
        %%%%%%%% Hardlimit
        x1=w'*x+b;
        y1=sign(x1);
    case {'gau'}
        %%%%%%%% Gaussian
        x1=w'*x+b;
        y1=exp(-x1.^2);
    case {'sig','sigmoid'}
        bias_vector = b*ones(1,size(x,2));
        %%%%%%%% Sigmoid 
        y1=1./(1+exp(-(w'*x+bias_vector)));
    case {'windows'}
        %%%%%%%% windows
        x1=w'*x+b;
        traina = x1<=1;
        trainb = x1>=-1;    
        y1 = traina.*trainb+0.0001;
        %%%%%%%% More activation functions can be added here
        
end
