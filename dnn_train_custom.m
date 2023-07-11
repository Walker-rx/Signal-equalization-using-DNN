function [ dlnet, velocity, losss, learnRate_save ] = dnn_train_custom(rate_times,maxEpochs, xTrain, yTrain, xValidation, yValidation, ...
                                                            numIterPerEpoch, miniBatchSize, dlnet, velocity, inilearningRate, ...
                                                            momentum, train_time, total_train_time, ...
                                                            LearnRateDropPeriod, LearnRateDropFactor, validationFrequency)
    ite = 0;
    learnRate = inilearningRate;
    losss = [];
    learnRate_save = [];
    averageGrad = [];
    averageSqGrad = [];
    gradDecay = 0.9;
    sqGradDecay = 0.999;

    valid_loss = 50000;
    valid_num = 0;
    valid_max = 100;
    XValidation(:,:,1) = xValidation;
    YValidation(:,:,1) = yValidation;
    dlXValidation = dlarray(single(XValidation),'CBT');
    dlYValidation = dlarray(single(YValidation),'CBT');
    dlXValidation = gpuArray(dlXValidation);
    dlYValidation = gpuArray(dlYValidation);

    for epoch = 1:maxEpochs
    
        for ite_loop = 1:numIterPerEpoch
            ite = ite + 1;

            %% Shuffling data
            if ite_loop==numIterPerEpoch
                idx = (ite_loop-1)*miniBatchSize+1 : numel(xTrain);
            else
                idx = (ite_loop-1)*miniBatchSize+1 : miniBatchSize*ite_loop;
            end
            xTrain_loop = xTrain(idx);
            yTrain_loop = yTrain(idx);

            xTrain_final = cell(1,rate_times);
            yTrain_final = cell(1,rate_times);
            colNum_perRate = size(xTrain{1},2)/rate_times;
            for i = 1:numel(xTrain_loop)
                for j = 1:rate_times
                    xTrain_final{j} = [ xTrain_final{j} xTrain_loop{i}( :,(j-1)*colNum_perRate+1:j*colNum_perRate ) ];
                    yTrain_final{j} = [ yTrain_final{j} yTrain_loop{i}( :,(j-1)*colNum_perRate+1:j*colNum_perRate ) ];
                end
            end
            for i = 1:numel(xTrain_final)
                colNum_tmp = size(xTrain_final{i},2);
                idx = randperm(colNum_tmp);
                xTrain_final{i} = xTrain_final{i}(:,idx);
                yTrain_final{i} = yTrain_final{i}(:,idx);
            end
            xTrain_final = cell2mat(xTrain_final);
            yTrain_final = cell2mat(yTrain_final);

            xTrain_final = gpuArray(xTrain_final);
            yTrain_final = gpuArray(yTrain_final);
            
            %% Training net
            X(:,:,1) = xTrain_final;
            Y(:,:,1) = yTrain_final;
            dlX = dlarray(single(X),'CBT');
            dlY = dlarray(single(Y),'CBT');
            dlX = gpuArray(dlX);
            dlY = gpuArray(dlY);
    
            [gradients,state,loss] = dlfeval(@modelGradientss,dlnet,dlX,dlY);
            dlnet.State = state;
    
%             [dlnet, velocity] = sgdmupdate(dlnet,gradients,velocity,learnRate,momentum);
            [dlnet, averageGrad,averageSqGrad] = adamupdate(dlnet,gradients,averageGrad,averageSqGrad,ite,learnRate,gradDecay,sqGradDecay);
    
            losss(ite) = extractdata(loss);
            learnRate_save(ite) = learnRate;
            if mod(ite,floor(numIterPerEpoch/5)) == 0
                fprintf(" training times = %d/%d , epoches = %d/%d , iteration = %d/%d , loss = %e , learnRate = %e \n",...
                    train_time,total_train_time,epoch,maxEpochs,ite,numIterPerEpoch,losss(ite),learnRate);
                pause(0.5)
            end
            clear X Y dlX dlY

            %% Validation           
            if mod(ite_loop,validationFrequency) == 0
                [~,~,loss_validation] = dlfeval(@modelGradientss,dlnet,dlXValidation,dlYValidation);
                loss_validation = gather(extractdata(loss_validation));
                if valid_loss <= loss_validation
                    valid_num = valid_num+1;
                    fprintf(" valid num = %d/%d , minimum loss = %e , valid num changed \n",valid_num,valid_max,valid_loss);
                else
                    valid_loss = loss_validation;
                    fprintf(" valid num = %d/%d , minimum loss = %e , valid num not changed \n",valid_num,valid_max,valid_loss);
                end
            end
            if valid_num == valid_max
                return
            end
        end
    
        if mod(epoch,LearnRateDropPeriod) == 0
            learnRate = learnRate*LearnRateDropFactor;
        end
    end

end