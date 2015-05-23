%function [TrainingTime, TrainingAccuracy, TestingAccuracy] = ...
function [TrainingTime, ConfusionMatrixTrain, ConfusionMatrixTest, CCRTrain, MSTrain, CCRTest, MSTest...
    	NumberofInputNeurons,NumberofHiddenNeurons,NumberofHiddenNeuronsFinal,NumberofOutputNeurons,InputWeight,OutputWeight,pstar_train] = ...
    kelm(train_data, test_data, Elm_Type, NumberofHiddenNeurons, ActivationFunction,  wMin, wMax,EE)

Kernel_type = EE.Kernel_type;
Kernel_para = EE.Kernel_para;% [1];
C= EE.C;% 0.1;


include_borders = EE.include_borders ;
include_k_clusters = EE.include_k_clusters ;



InputWeight = -1; 


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
n = size(T,2);
NumberofHiddenNeuronsFinal = n;



Kernelsextended = P';

if(include_borders==1)
    %size(eye(size(XTrain,2)))
   % XTrain
   Kernelsextended = [Kernelsextended; 2*eye(size(XTrain,2))-1] 
   n=n+ size(XTrain,2);
end



Omega_train = kernel_matrix(P',Kernelsextended,Kernel_type, Kernel_para);
OutputWeight=((Omega_train+speye(n)/C)\(T')); 
TrainingTime=toc;

%%%%%%%%%%% Calculate the training output
Y=(Omega_train * OutputWeight)'; 




%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
%OutputWeight=pinv(H') * T';                        % slower implementation
TrainingTime = toc(tStart);

%%%%%%%%%%% Calculate the training accuracy
%Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data





if Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y));               %   Calculate training accuracy (RMSE) for regression case
end
clear H;

%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;

Omega_test = kernel_matrix(P',Kernelsextended,Kernel_type, Kernel_para,TV.P');
TY=(Omega_test' * OutputWeight)';                            %   TY: the actual output of the testing data

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
    





function omega = kernel_matrix(Xtrain,Kernelsextended,kernel_type, kernel_pars,Xt)

nb_data = size(Kernelsextended,1);


if strcmp(kernel_type,'RBF_kernel'),
    if nargin<5,
        XXh = sum(Kernelsextended.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega./kernel_pars(1));
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*Xtrain*Xt';
        omega = exp(-omega./kernel_pars(1));
    end
    
elseif strcmp(kernel_type,'lin_kernel')
    if nargin<5,
        omega = Xtrain*Xtrain';
    else
        omega = Xtrain*Xt';
    end
    
elseif strcmp(kernel_type,'poly_kernel')
    if nargin<5,
        omega = (Xtrain*Xtrain'+kernel_pars(1)).^kernel_pars(2);
    else
        omega = (Xtrain*Xt'+kernel_pars(1)).^kernel_pars(2);
    end
    
elseif strcmp(kernel_type,'wav_kernel')
    if nargin<5,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        
        XXh1 = sum(Xtrain,2)*ones(1,nb_data);
        omega1 = XXh1-XXh1';
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
        
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*(Xtrain*Xt');
        
        XXh11 = sum(Xtrain,2)*ones(1,size(Xt,1));
        XXh22 = sum(Xt,2)*ones(1,nb_data);
        omega1 = XXh11-XXh22';
        
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
    end
end

