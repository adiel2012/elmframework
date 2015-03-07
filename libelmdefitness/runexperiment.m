function runexperiment(EE)

train_data = load(EE.dbPreffixTrain);
test_data = load(EE.dbPreffixTest);

%%%%%%%%%%% Load training dataset
T=train_data(:,1);
P=train_data(:,2:size(train_data,2));

%%%%%%%%%%% Load testing dataset
TV.T=test_data(:,1);
TV.P=test_data(:,2:size(test_data,2));

%%%%%%%%%%% Check for constant attributes that we can delete. Otherwise a
%%%%%%%%%%% NaN can be obtained later.
minvals = min(P);
maxvals = max(P);

r = 0;
for k=1:size(P,2)
    if minvals(k) == maxvals(k)
        r = r + 1;
        index(r) = k;
    end
end

if r > 0
	r = 0;
	for k=1:size(index,2)
	    P(:,index(k)-r) = [];
	    TV.P(:,index(k)-r) = [];
	    r = r + 1;
	end
end

clear index r minvals minvals;

% Process data: standarize, normalize, scale

switch lower(EE.preprocessData)
    case {'scale'}
        [P,TV.P] = scale(P,TV.P, EE.XVMin, EE.XVMax);
        %P = P';
        %TV.P = TV.P';
    otherwise
        disp('ERROR: Unknown preprocess data method. ')
        exit;
end

train_data = [T P];
test_data = [TV.T TV.P];

% Open a matlabpool
if EE.multicore
    isOpen = matlabpool('size') > 0;
    if ~isOpen
        matlabpool(EE.cores);
    end
end

%% ORIGINAL E-ELM
if (EE.eelm)
    disp(['ORIGINAL E-ELM']);
    EE.elmAlgorithm = 'eelm';
    % Calculate the optimal number of hidden nodes
    bestHnodes = calculate_optimal_hiddenn(EE,train_data);
    disp(['BEST HIDDEN NODES: ' num2str(bestHnodes.nhidden)]);   
    EE.nhidden = bestHnodes.nhidden; 
    logLevel = 2;
    
    
    % Perform the algorithm
    eelm_experiment(EE,train_data,test_data,logLevel);
end

%% TODO: unhack
EE.multicore = 0;
sizeData = size(train_data,1);

%% OP-ELM
if (EE.opelm)
    
    disp('OP-ELM');    
    % Determine the maximum number of hidden nodes (OP-ELM paper)
    Nmax = round(sizeData - sizeData*0.1) - 4;
    
    if (EE.nhidden > Nmax - 5)
       EE.nhidden =  Nmax - 5;
    end;

    EE.elmAlgorithm = 'opelm';
    EE.preprocessData = 'n';
    
    logLevel = 2;
    eelm_experiment(EE,train_data,test_data,logLevel);

end

%% ORIGINAL ELM
if (EE.elm)
    
    disp(['ORIGINAL ELM']);
    EE.elmAlgorithm = 'elm';
    
    bestHnodes = calculate_optimal_hiddenn(EE,train_data);
    disp(['BEST HIDDEN NODES: ' num2str(bestHnodes.nhidden)]);   
    EE.nhidden = bestHnodes.nhidden;        
    logLevel = 2;
    eelm_experiment(EE,train_data,test_data,logLevel);
      
end

%% PCA-ELM
if (EE.pcaelm)
    
    disp(['PCA-ELM']);
    EE.elmAlgorithm = 'pcaelm';
    logLevel = 2;
    eelm_experiment(EE,train_data,test_data,logLevel);
      
end

%% LDA-ELM
if (EE.ldaelm)
    
    disp(['LDA-ELM']);
    EE.elmAlgorithm = 'ldaelm';
    logLevel = 2;
    eelm_experiment(EE,train_data,test_data,logLevel);
      
end

%% PCA-LDA-ELM
if (EE.pcaldaelm)
    
    disp(['PCA-LDA-ELM']);
    EE.elmAlgorithm = 'pcaldaelm';
    logLevel = 2;
    eelm_experiment(EE,train_data,test_data,logLevel);
      
end

%% I-ELM
if (EE.ielm)
    
    disp(['I-ELM']);
    EE.elmAlgorithm = 'ielm';
    
   
    logLevel = 2;
    eelm_experiment(EE,train_data,test_data,logLevel);
      
end

%% P-ELM
if (EE.pelm)
    
    disp(['P-ELM']);
    EE.elmAlgorithm = 'pelm';
    
   
    logLevel = 2;
    eelm_experiment(EE,train_data,test_data,logLevel);
      
end

%% EM-ELM
if (EE.em_elm)
    
    disp(['EM-ELM']);
    EE.elmAlgorithm = 'em_elm';
    
   
    logLevel = 2;
    eelm_experiment(EE,train_data,test_data,logLevel);
      
end

%% Cerrar piscina de hebras de matlab
if EE.multicore
 isOpen = matlabpool('size') > 0;
 if ~isOpen
     matlabpool close;
 end
end

clear EE train_data test_data logLevel;
