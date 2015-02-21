function bestHiddenNodes  = calculate_optimal_hiddenn(EE,train_data)
    % Only log cross validation results
    logLevel = 1;
    EE.repeatfold = 3 ;
    EE.itermax = 3;
    %crossValMulticore = EE.multicore; TODO: fix cross-validation parfor and cell issues. 
    crossValMulticore = 0;
    EE.multicore = 0;
    
    Y = train_data(:,1);
    data = train_data(:,2:size(train_data,2));
    % Get 10-fold from training data
    CVO = cvpartition(Y,'k',10);
    
    EEFoldStats = cell(size(EE.hNodesSet,2),CVO.NumTestSets);

    CCRtemp = zeros(CVO.NumTestSets,1);
    MStemp = zeros(CVO.NumTestSets,1);
    
    bestIdx = [0 0];
    bestFitness = 0;
    
    train = cell(CVO.NumTestSets);
    test = cell(CVO.NumTestSets);
    
    % Build the folds now for better paralelization
    for jj = 1:CVO.NumTestSets
        trIdx = CVO.training(jj);
        teIdx = CVO.test(jj);
        train{jj} = [Y(trIdx,:) data(trIdx,:)];
        test{jj} = [Y(teIdx,:) data(teIdx,:)];
    end
            
    % Foreach hidden nodes value...
    for ii = 1:size(EE.hNodesSet,2)
        EE.nhidden = EE.hNodesSet(ii);
        %lambdaStats = cell(CVO.NumTestSets,1);
        
        % Foreach fold
        if crossValMulticore == 1
            parfor jj = 1:CVO.NumTestSets
                EEFoldStats{ii,jj} = eelm_experiment(EE,train{jj},test{jj},logLevel);

                CCRtemp(jj) = EEFoldStats{ii,jj}.CCRTest_mean;
                MStemp(jj) = EEFoldStats{ii,jj}.MSTest_mean;
            end
        else
            for jj = 1:CVO.NumTestSets
                EEFoldStats{ii,jj} = eelm_experiment(EE,train{jj},test{jj},logLevel);

                CCRtemp(jj) = EEFoldStats{ii,jj}.CCRTest_mean;
                MStemp(jj) = EEFoldStats{ii,jj}.MSTest_mean;
            end
        end
        
        % TODO: Consider standard deviation?
        CCRmeanIt = mean(CCRtemp);
        CCRstdIt = std(CCRtemp);
        MSmeanIt = mean(MStemp);
        MSstdIt = std(MStemp);
        
        fitnessIt = CCRmeanIt;

        if fitnessIt > bestFitness 
            bestIdx = ii;
            bestFitness = fitnessIt;
            CCRmean_best = CCRmeanIt;
            CCRstd_best = CCRstdIt;
            MSmean_best = MSmeanIt;
            MSstd_best = MSstdIt;
        end 

    end
    
    bestHiddenNodes.nhidden = EEFoldStats{bestIdx,1}.nhidden;
    bestHiddenNodes.CCRTest_mean = CCRmean_best;
    bestHiddenNodes.CCRTest_std = CCRstd_best;
    bestHiddenNodes.MSTest_mean = MSmean_best;
    bestHiddenNodes.MSTest_std = MSstd_best;

    clear CCRmean_best CCRstd_best MSmean_best MSstd_best;
    clear CVO EEFoldStats train_data CCRtemp MStemp;
      
