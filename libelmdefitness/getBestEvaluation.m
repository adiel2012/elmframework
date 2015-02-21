function [OutputWeight,CCRTrain, MSTrain,CCRTest, MSTest] = ...
    getBestEvaluation (weight_bias, P, T, TV, T_foo, TV_foo, NumberofHiddenNeurons,ActivationFunction,OutputWeight)

% Esto ya se ha calculado antes y se podría reutilizar
NumberofInputNeurons=size(P, 1);
NumberofTrainingData=size(P, 2);
NumberofTestingData=size(TV.P, 2);

temp_weight_bias=reshape(weight_bias, NumberofHiddenNeurons, NumberofInputNeurons+1);
InputWeight=temp_weight_bias(:, 1:NumberofInputNeurons);

switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        Gain=1;
        BiasofHiddenNeurons=temp_weight_bias(:,NumberofInputNeurons+1);
        tempH=InputWeight*P;
        ind=ones(1,NumberofTrainingData);
        BiasMatrix=BiasofHiddenNeurons(:,ind);      %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
        tempH=tempH+BiasMatrix;
        clear BiasMatrix
        H = 1 ./ (1 + exp(-Gain*tempH));
        clear temp_weight_bias ind BiasMatrix tempH;
        
        tempH_test=InputWeight*TV.P;
        ind=ones(1,NumberofTestingData);
        BiasMatrix=BiasofHiddenNeurons(:,ind);      %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
        tempH_test=tempH_test + BiasMatrix;
        H_test = 1 ./ (1 + exp(-Gain*tempH_test));
        
        clear Gain BiasofHiddenNeurons ind BiasMatrix tempH_test;
        
    case {'up'}
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
    
        H_test = zeros(NumberofHiddenNeurons,NumberofTestingData);
        for i = 1 : NumberofTestingData
            for j = 1 : NumberofHiddenNeurons
                temp = zeros(NumberofInputNeurons,1);
                for n = 1: NumberofInputNeurons
                    temp(n) = InputWeight(j,n)*TV.P(n,i);
                end
                H_test(j,i) =  sum(temp);
            end
        end

        clear temp;
        
    case {'rbf'}
        P = P';
        W10 = temp_weight_bias(:,NumberofInputNeurons+1)';
        W1 = InputWeight;
        % TODO: Un hack
        H = zeros(NumberofTrainingData,NumberofHiddenNeurons);
        for j=1:NumberofHiddenNeurons
            H(:,j)=gaussian_func(P,W1(j,:),W10(1,j));
            %KM.valueinit(:,j)=gaussian_func(x,W1(j,:),W10(1,j));
        end
        P = P';
        H = H';
        
        TV.P = TV.P';

        % TODO: Un hack
        H_test = zeros(NumberofTestingData,NumberofHiddenNeurons);
        for j=1:NumberofHiddenNeurons
            H_test(:,j)=gaussian_func(TV.P,W1(j,:),W10(1,j));
            %KM.valueinit(:,j)=gaussian_func(x,W1(j,:),W10(1,j));
        end
        TV.P = TV.P';
        H_test = H_test';
end

clear NumberofInputNeurons NumberofHiddenNeurons NumberofTrainingData NumberofTestingData;

Y=(H' * OutputWeight)';      
TY=(H_test' * OutputWeight)';
    
clear H H_test;

[winnerTrain LabelTrainPredicted] = max(Y);
[winnerTest LabelTestPredicted] = max(TY);

LabelTrainPredicted = LabelTrainPredicted';
LabelTrainPredicted = LabelTrainPredicted -1;
LabelTestPredicted = LabelTestPredicted';
LabelTestPredicted = LabelTestPredicted -1;

ConfusionMatrixTrain = confmat(T_foo',LabelTrainPredicted);
ConfusionMatrixTest = confmat(TV_foo',LabelTestPredicted);

CCRTrain = CCR(ConfusionMatrixTrain);
CCRTest = CCR(ConfusionMatrixTest);

MSTrain = Sensitivity(ConfusionMatrixTrain);
MSTest = Sensitivity(ConfusionMatrixTest);

% switch lower(FitnessFunction)
%     case {'ccr'}
%         Fitness = CCRTrain;
%     case {'ccrs'}
%         Fitness = (1-lambdaWeigth)*CCRTrain + lambdaWeigth*MSTrain;
%     case {'ccrs_c'}
%         % TODO
%         Fitness = (1-lambdaWeigth)*CCRTrain + lambdaWeigth*MSTrain;
% end

Fitness = -1;

clear LabelTrainPredicted LabelTestPredicted ConfusionMatrixTrain ConfusionMatrixTest winnerTrain winnerTest;
