% close all
% clear
% inputSize = 30;
% numHiddenUnits = 6;
% outputSize = 1;
% layers = [...
%     sequenceInputLayer(inputSize)
%     fullyConnectedLayer(numHiddenUnits)
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
% lgraph = layerGraph(layers);
clear
inputSize = 10;
miniBatchSize = 32;

layers = [
    sequenceInputLayer(inputSize)
    % fullyConnectedLayer(20)
    % reluLayer()
    % fullyConnectedLayer(1)
    regressionLayer()
];

net = assembleNetwork(layers);

% 查看输入和输出的维度
inputSize1 = net.Layers(1).InputSize;  % 输入层的维度
% outputSize = net.Layers(end).OutputSize;  % 输出层的维度

% disp("输入层的维度: " + num2str(inputSize));
% disp("输出层的维度: " + num2str(outputSize));
