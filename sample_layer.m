classdef sample_layer < nnet.layer.Layer
    
    properties

    end

    methods
        function layer = sample_layer(numoutputs,name)
            layer.NumOutputs = numoutputs;           
            layer.Name = name;
        end

        function [Z1,Z2] = predict(layer,sam_input)
            % rowNum = size(X,1);
            % split_rowNum = rowNum/2;
            % Z = mat2cell(X,ones(1,2)*split_rowNum);
            % [Z1,Z2] = Z{:};
            colNum = size(sam_input,2);
            split_colNum = colNum/layer.NumOutputs;
            if colNum == 1
                Z1=sam_input;
                Z2=sam_input;
            else
                for i = 1:layer.NumOutputs
                    eval(['Z',num2str(i),'=sam_input(:,',num2str((i-1)*split_colNum+1),':',num2str(i*split_colNum),');']);
                end
            end
        end
    end
end