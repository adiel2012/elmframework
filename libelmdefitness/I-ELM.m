function [TrainingTime, TrainingAccuracy, TestingAccuracy] = I-ELM(TrainingData_File, TestingData_File, MaxNumberofHiddenNeurons, ActivationFunction, Problem_Type)

% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% MaxNumberofHiddenNeurons - Maximum number of hidden neurons assigned
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%                           'rbf' for Gaussian function (division)
%                           'rbf_gamma' for Gaussian function (product)

% Problem_Type          - 0 for regression; 1 for classification

% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression or correct classification rate for classification
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression or correct classification rate for classification

% Sample1 regression: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm('sinc_train', 'sinc_test', 0, 20, 'sig')
% Sample2 classification: elm('diabetes_train', 'diabetes_test', 1, 20, 'sig')

%obtain the function handle and execute the function by using feval

                                        %%%%    Authors: DR. HUANG GUANGBIN and Mr. Chen Lei 
                                        %%%%    NANYANG TECHNOLOGICAL UNIVERSITY
                                        %%%%    EMAIL: EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
                                        %%%%    DATE: July 2006
REGRESSION = 0;
CLASSIFIER = 1;
%%%%%%%%%%% Load training dataset
train_data=load(TrainingData_File);
T=train_data(:,1)';
I=train_data(:,2:size(train_data,2))';
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
test_data=load(TestingData_File);
t_testing=test_data(:,1)';
x_testing=test_data(:,2:size(test_data,2))';
clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(I,2);
NumberofTestingData=size(x_testing,2);
NumberofInputNeurons=size(I,1);
NumberofOutputNeurons=size(T,1);

if Problem_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,t_testing),2);
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
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T = temp_T*2-1;

    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == t_testing(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    t_testing = temp_TV_T*2-1;
end  
clear temp_T;

L=0;    % L: number of neurons;

TrainingResidualError=ones(size(T));

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

while L<MaxNumberofHiddenNeurons
    
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

    HiddenNeuronActivation(:,L)=hidden_output(I,InputWeight(:,L),HiddenBias(L),ActivationFunction,NumberofInputNeurons)';

    for i=1:NumberofOutputNeurons
        SumofResidualHiddenActivation(i,:)=TrainingResidualError(i,:)*HiddenNeuronActivation(:,L);
        SumofSquareHiddenActivation=HiddenNeuronActivation(:,L)'*HiddenNeuronActivation(:,L);    
        Beta(i,L)=SumofResidualHiddenActivation(i,:)/SumofSquareHiddenActivation;
    end       

    TrainingResidualError_Previous=TrainingResidualError;   % Record (L-1)th residual error
    TrainingResidualError=TrainingResidualError_Previous-Beta(:,L)*HiddenNeuronActivation(:,L)';   % Calculate L-th residual error
    if Problem_Type==REGRESSION
        TrainingResidualError_Norm(L)=norm(TrainingResidualError)/sqrt(NumberofTrainingData);
    end

    total_cputimesofar(L)=double(cputime-starting_cpu); % CPU time spent for training
    
    for n=1:NumberofTestingData   % Calculate testing residual error after k-th neuron added. 
        TestingResidualError(:,n)= t_testing(:,n);
    end
    HiddenNeuronActivationTesting(:,L)=hidden_output(x_testing,InputWeight(:,L),HiddenBias(L),ActivationFunction,NumberofInputNeurons)';

    TestingResidualError_Previous=TestingResidualError;   % Record (L-1)th residual error
    TestingResidualError=TestingResidualError_Previous-Beta(:,L)*HiddenNeuronActivationTesting(:,L)';   % Calculate L-th testing residual error
    if Problem_Type==REGRESSION
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

    HiddenNeuronActivation(:,L)=hidden_output(I,InputWeight(:,L),HiddenBias(L),ActivationFunction,NumberofInputNeurons)';

    for i=1:NumberofOutputNeurons
        SumofResidualHiddenActivation(i,:)=TrainingResidualError(i,:)*HiddenNeuronActivation(:,L);
        SumofSquareHiddenActivation=HiddenNeuronActivation(:,L)'*HiddenNeuronActivation(:,L);    
        Beta(i,L)=SumofResidualHiddenActivation(i,:)/SumofSquareHiddenActivation;
    end      

    TrainingResidualError_Previous=TrainingResidualError;   % Record (L-1)th residual error
    TrainingResidualError=TrainingResidualError_Previous-Beta(:,L)*HiddenNeuronActivation(:,L)';   % Calculate L-th residual error
    if Problem_Type==REGRESSION
        TrainingResidualError_Norm(L)=norm(TrainingResidualError)/sqrt(NumberofTrainingData);    
    end

    total_cputimesofar(L)=total_cputimesofar(L-1)+cputime-starting_cpu; % CPU time spent for training
    
    HiddenNeuronActivationTesting(:,L)=hidden_output(x_testing,InputWeight(:,L),HiddenBias(L),ActivationFunction,NumberofInputNeurons)';
    TestingResidualError_Previous=TestingResidualError;   % Record (L-1)th residual error
    TestingResidualError=TestingResidualError_Previous-Beta(:,L)*HiddenNeuronActivationTesting(:,L)';   % Calculate L-th testing residual error
    if Problem_Type==REGRESSION
        TestingResidualError_Norm(L)=norm(TestingResidualError)/sqrt(NumberofTestingData);
    end
end        

end % End while when TrainingResidualError not larger than min_goal

TrainingTime=total_cputimesofar;

if Problem_Type == REGRESSION
TrainingAccuracy=TrainingResidualError_Norm;
TestingAccuracy=TestingResidualError_Norm;
end

if Problem_Type == CLASSIFIER
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
 
        for i = 1 : size(t_testing, 2)
            [x, label_index_expected]=max(t_testing(:,i));
            [x, label_index_actual]=max(Out_test(:,i));
            if label_index_actual~=label_index_expected
                MissClassificationRate_Testing=MissClassificationRate_Testing+1;
            end
        end
        TestingAccuracy(k) = 1-MissClassificationRate_Testing/size(t_testing,2);
    end
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
