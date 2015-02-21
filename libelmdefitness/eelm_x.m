function [Fitness,OutputWeight] = eelm_x (Elm_Type, weight_bias, P, T, T_org, T_ceros, NumberofHiddenNeurons,ActivationFunction)

NumberofInputNeurons=size(P, 1);
NumberofTrainingData=size(P, 2);
NumberofOutputNeurons=size(T, 1);
%NumberofTestingData=size(TV.P, 2);

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
        H = 1 ./ (1 + exp(-Gain*tempH));
        %clear temp_weight_bias Gain BiasofHiddenNeurons ind BiasMatrix tempH;
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
        %InputWeight = [W1 W10'];
end

OutputWeight=pinv(H') * T';
Y=(H' * OutputWeight)';

clear H;

if Elm_Type == 1
    
    [winnerTrain LabelTrainPredicted] = max(Y);

    LabelTrainPredicted = LabelTrainPredicted';
    LabelTrainPredicted = LabelTrainPredicted -1;        

    ConfusionMatrixTrain = confmat(T_org',LabelTrainPredicted);

    CCRTrain = CCR(ConfusionMatrixTrain);
    Fitness = 1-CCRTrain; 
    
    
else % regression
    Fitness=sqrt(mse(T - Y));
end

clear NumberofInputNeurons NumberofHiddenNeurons NumberofTrainingData
