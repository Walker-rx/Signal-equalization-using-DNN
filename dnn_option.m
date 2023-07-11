% inputSize = 60;
numHiddenUnits = 60;
outputSize = 1;
rate_times = 2;
hiddlayer_num = 3;

layers = [...
    sequenceInputLayer(inputSize)
    sample_layer(rate_times,'sam_layer')    
];
lgraph = layerGraph(layers);
mergeLayer_name = 'mergeLayer';
mergeLayer = merge_layer(rate_times,mergeLayer_name);
lgraph = addLayers(lgraph,mergeLayer);
for i = 1:rate_times

    fullLayer_name = ['fc',num2str(i),'_1'];
    reluLayer_name = ['relu',num2str(i),'_1'];
    full_Layer = fullyConnectedLayer(numHiddenUnits,'Name',fullLayer_name);
    relu_Layer = reluLayer('Name',reluLayer_name);
    lgraph = addLayers(lgraph,full_Layer);
    lgraph = addLayers(lgraph,relu_Layer);
    lgraph = connectLayers(lgraph,['sam_layer/out',num2str(i)],fullLayer_name);
    lgraph = connectLayers(lgraph,fullLayer_name,reluLayer_name);

    for j = 2:hiddlayer_num
        reluLayer_prename = ['relu',num2str(i),'_',num2str(j-1)];
        fullLayer_name = ['fc',num2str(i),'_',num2str(j)];
        reluLayer_name = ['relu',num2str(i),'_',num2str(j)];
        full_Layer = fullyConnectedLayer(numHiddenUnits,'Name',fullLayer_name);
        relu_Layer = reluLayer('Name',reluLayer_name);
        lgraph = addLayers(lgraph,full_Layer);
        lgraph = addLayers(lgraph,relu_Layer);
        lgraph = connectLayers(lgraph,reluLayer_prename,fullLayer_name);
        lgraph = connectLayers(lgraph,fullLayer_name,reluLayer_name);
    end

    lgraph = connectLayers(lgraph,reluLayer_name,[mergeLayer_name,'/in',num2str(i)]);
    % reluLayer_prename = ['relu',num2str(i),'_',num2str(hiddlayer_num)];
    % fullLayer_name = ['fc',num2str(i),'_',num2str(hiddlayer_num+1)];
    % outLayer_name = ['outLayer',num2str(i)];
    % full_Layer = fullyConnectedLayer(outputSize,'Name',fullLayer_name);
    % outLayer = regressionLayer('Name',outLayer_name);    
    % lgraph = addLayers(lgraph,full_Layer);
    % lgraph = addLayers(lgraph,outLayer);
    % lgraph = connectLayers(lgraph,reluLayer_prename,fullLayer_name);
    % lgraph = connectLayers(lgraph,fullLayer_name,outLayer_name);

end
fullLayer_name = ['fc','_',num2str(hiddlayer_num+1)];
outLayer_name = 'outLayer';
full_Layer = fullyConnectedLayer(outputSize,'Name',fullLayer_name);
outLayer = regressionLayer('Name',outLayer_name); 
lgraph = addLayers(lgraph,full_Layer);
% lgraph = addLayers(lgraph,outLayer);
lgraph = connectLayers(lgraph,mergeLayer_name,fullLayer_name);
% lgraph = connectLayers(lgraph,fullLayer_name,outLayer_name);
plot(lgraph);

% layers = [...
%     sequenceInputLayer(inputSize)
% ];
% lgraph = layerGraph(layers);
% lgraph = layerGraph;
% input_layer = sequenceInputLayer(inputSize,'Name','inputLayer');
% lgraph = addLayers(lgraph,input_layer);
% fullLayer_name_1 = 'fc1_1';
% full_Layer_1 = fullyConnectedLayer(numHiddenUnits,'Name',fullLayer_name_1);
% fullLayer_name_2 = 'fc1_2';
% full_Layer_2 = fullyConnectedLayer(numHiddenUnits,'Name',fullLayer_name_2);
% fullLayer_name_3 = 'fc2_1';
% full_Layer_3 = fullyConnectedLayer(numHiddenUnits,'Name',fullLayer_name_3);
% fullLayer_name_4 = 'fc2_2';
% full_Layer_4 = fullyConnectedLayer(numHiddenUnits,'Name',fullLayer_name_4);
% relu1 = reluLayer('Name','relu_1');
% relu2 = reluLayer('Name','relu_2');
% % lgraph = addLayers(lgraph,full_Layer_1);
% % lgraph = addLayers(lgraph,full_Layer_2);
% % lgraph = addLayers(lgraph,full_Layer_3);
% % lgraph = addLayers(lgraph,full_Layer_4);
% lgraph = addLayers(lgraph,relu1);
% lgraph = addLayers(lgraph,relu2);
% sam_layer=sample_layer(2,'sam_layer');
% lgraph = addLayers(lgraph,sam_layer);
% lgraph = connectLayers(lgraph,'inputLayer','sam_layer');
% lgraph = connectLayers(lgraph,'sam_layer/out1','relu_1');
% lgraph = connectLayers(lgraph,'sam_layer/out2','relu_2');
% % lgraph = connectLayers(lgraph,'relu_1','fc1_1');
% % lgraph = connectLayers(lgraph,'relu_2','fc2_1');
% % lgraph = connectLayers(lgraph,'fc1_1','fc1_2');
% % lgraph = connectLayers(lgraph,'fc2_1','fc2_2');
% mergeLayer_name = 'mergeLayer';
% mergeLayer = merge_layer(rate_times,mergeLayer_name);
% lgraph = addLayers(lgraph,mergeLayer);
% lgraph = connectLayers(lgraph,'relu_1',[mergeLayer_name,'/in1']);
% lgraph = connectLayers(lgraph,'relu_2',[mergeLayer_name,'/in2']);
% % lgraph = connectLayers(lgraph,'fc1_2',[mergeLayer_name,'/in1']);
% % lgraph = connectLayers(lgraph,'fc2_2',[mergeLayer_name,'/in2']);
% % fullLayer_name = ['fc','_end'];
% outLayer_name = 'outLayer';
% % full_Layer = fullyConnectedLayer(outputSize,'Name',fullLayer_name);
% outLayer = regressionLayer('Name',outLayer_name); 
% % lgraph = addLayers(lgraph,full_Layer);
% lgraph = addLayers(lgraph,outLayer);
% % lgraph = connectLayers(lgraph,mergeLayer_name,fullLayer_name);
% % lgraph = connectLayers(lgraph,fullLayer_name,outLayer_name);
% lgraph = connectLayers(lgraph,mergeLayer_name,outLayer_name);
% plot(lgraph);
% layers = [...
%     sequenceInputLayer(inputSize)
%     preluLayer(20,'prelu')
%     sample_layer(1,'sam_layer')
%     fullyConnectedLayer(numHiddenUnits)
%     reluLayer % 1
%     fullyConnectedLayer(numHiddenUnits)
%     reluLayer % 2
%     fullyConnectedLayer(numHiddenUnits)
%     sigmoidLayer % 3
%     fullyConnectedLayer(numHiddenUnits)
%     reluLayer % 4
%     fullyConnectedLayer(numHiddenUnits)
%     reluLayer % 5
%     fullyConnectedLayer(outputSize)
%     regressionLayer];


options = trainingOptions('adam', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',LearnRateDropFactor,...
    'LearnRateDropPeriod',LearnRateDropPeriod,...
    'ValidationData',{xValidation,yValidation},...
    'ValidationFrequency',validationFrequency,...
    'ValidationPatience',80,...
    'Verbose',true,...
    'InitialLearnRate',inilearningRate,...
    'ExecutionEnvironment','gpu');
%         'Plots','training-progress');
% 'ExecutionEnvironment','gpu',...