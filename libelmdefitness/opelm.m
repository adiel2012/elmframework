function [TrainingTime, ConfusionMatrixTrain, ConfusionMatrixTest, CCRTrain, MSTrain, CCRTest, MSTest...
    	NumberofInputNeurons,NumberofHiddenNeurons,NumberofHiddenNeuronsFinal,NumberofOutputNeurons,InputWeight,OutputWeight] = ...
            opelm(train_data, test_data, Elm_Type, NumberofHiddenNeurons, ActivationFunction)

%% PROCESS DATA

data.y=train_data(:,1);
data.x=train_data(:,2:size(train_data,2));
clear train_data;

data_t.y=test_data(:,1);
data_t.x=test_data(:,2:size(test_data,2));
clear test_data;

data.y=data.y';
data.x=data.x';

data_t.y=data_t.y';
data_t.x=data_t.x';

NumberofTrainingData=size(data.x,2);
NumberofTestingData=size(data_t.x,2);
NumberofInputNeurons=size(data.x,1);

%%%%%%%%%%%% Preprocessing the data of classification
sorted_target=sort(cat(2,data.y,data_t.y),2);
label=zeros(1,1);
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
        if label(1,j) == data.y(1,i)
            break; 
        end
    end
    temp_T(j,i)=1;
end
data.y=temp_T*2-1;

%%%%%%%%%% Processing the targets of testing
temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
for i = 1:NumberofTestingData
    for j = 1:number_class
        if label(1,j) == data_t.y(1,i)
            break; 
        end
    end
    temp_TV_T(j,i)=1;
end
data_t.y=temp_TV_T*2-1;

clear temp_T temp_TV_T;

data.y=data.y';
data.x=data.x';
data_t.y=data_t.y';
data_t.x=data_t.x';

%% OP-ELM EXPERIMENT
%kernel = ActivationFunction;
%maxneur = 500;
%problem = 'c';
%normal= 'y';

switch lower(ActivationFunction)
    case {'sig'}
        kernel = 's';
    case {'rbf'}
        kernel = 'g';
end

if Elm_Type == 0
    problem = 'r';
else
    problem = 'c';
end


%%%%%%%%%%%%%%%%%%%%%%% UNHACK
kernel = 'lsg';
%%%%%%%%%%%%%%%%%%%%%%% UNHACK

% The data are already preprocesed
normal = 'n';

% Train the model
[model,TrainingTime]=train_OPELM(data,kernel,NumberofHiddenNeurons,problem,normal);

clear problem normal;

if isempty(model)
    disp('Error in OPELM')
    return;
end

minvals = min(model.yh');
maxvals = max(model.yh');

r = 0;
for k=1:size(model.yh,1)
    if minvals(k) == maxvals(k)
        r = r + 1;
        index(r) = k;
    end
end

if r > 0
    disp('Error is coming')
end


% TEST
[yh,error]=sim_OPELM(model,data_t);
    

% rmse = sqrt(mse(data_t.y - yh))

minvals = min(yh');
maxvals = max(yh');

r = 0;
for k=1:size(yh,1)
    if minvals(k) == maxvals(k)
        r = r + 1;
        index(r) = k;
    end
end

if r > 0
    disp('Error is coming')
end



% Calculate CCR and MS
label_y = zeros(size(model.y,2),1);
label_yh = zeros(size(model.yh,2),1);
label_t_y = zeros(size(data_t.y,2),1);
label_t_yh = zeros(size(data_t.y,2),1);

% Label format conversion
for j = 1:number_class
    index = model.y(:,j)==1;
    label_y(index) = j;
    index = model.yh(:,j)==1;
    label_yh(index) = j;
    
    index = data_t.y(:,j)==1;
    label_t_y(index) = j;
    index = yh(:,j)==1;
    label_t_yh(index) = j;
end

ConfusionMatrixTrain = confmat(label_y,label_yh);
ConfusionMatrixTest = confmat(label_t_y,label_t_yh);

clear data data_t label_y label_yh label_t_y label_t_yh;

CCRTrain = CCR(ConfusionMatrixTrain)*100;
CCRTest = CCR(ConfusionMatrixTest)*100;
MSTrain = Sensitivity(ConfusionMatrixTrain)*100;
MSTest = Sensitivity(ConfusionMatrixTest)*100;

% TODO: EXTRACT OP-ELM structure
InputWeight = [model.KM.param.p1' model.KM.param.p2'];
OutputWeight = model.W2;

NumberofHiddenNeuronsFinal = size(OutputWeight,1)-1;

