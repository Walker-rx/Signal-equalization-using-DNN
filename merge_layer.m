classdef merge_layer < nnet.layer.Layer
    methods
        function layer = merge_layer(numInputs,name) 
            layer.NumInputs = numInputs;
            layer.Name = name;
        end
        
        function Z = predict(layer,varargin)
            Z = [];
            X = varargin;
            for i = 1:layer.NumInputs
                Z = [Z,X{i}];
            end
        end
    end
end