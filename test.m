clear
inputSize = 10;
miniBatchSize = 32;

layers = [
    sequenceInputLayer(inputSize)
    fullyConnectedLayer(20)
    reluLayer()
    fullyConnectedLayer(1)
];
lgraph = layerGraph(layers);
plot(lgraph)
dlnet = dlnetwork(lgraph);
if exist("dlnet")
    a = 10;
end