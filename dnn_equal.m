clear
close all
tStart = tic;

ver = 1;

t = datetime('now');
folder = '6.18';
ori_rate = 30e6;
rec_rate = 60e6;
% load_path_ini = "/home/xliangseu/ruoxu/equalization-using-DNN/data_save/light_data_"+folder;
load_path_ini = "/home/oem/Users/ruoxu/equalization-using-DNN/data_save/light_data_"+folder;
save_path_ini = "data_save/light_data_"+folder;
save_path = save_path_ini + "/dnn/"+t.Month+"."+t.Day+"/mix_bias_amp/Threenonlinear"+ver;   

if(~exist(save_path,'dir'))
    mkdir(char(save_path));
end

%% Network parameters
% bias_scope = 0.05:0.04:0.85;
% amp_scope_ini = [0.005 0.007 0.015 0.024 0.034 0.045 0.08 0.18 0.25 0.3];
bias_scope = 0.45;
amp_scope_ini = 0.18;

total_cell = 60;
total_data_num = total_cell;

loop_data_num = ceil(21*50/(length(bias_scope)*length(amp_scope_ini)))+1;
if loop_data_num>30
    loop_data_num = 30;
end
loop_train_num = ceil(total_cell/loop_data_num);

data_scope = cell(1,loop_train_num);
for i = 1:loop_train_num
    if i == loop_train_num
        data_scope{i} = [(i-1)*loop_data_num+1 , total_cell];
    else
        data_scope{i} = [(i-1)*loop_data_num+1 , i*loop_data_num];
    end   
end

total_loop_time = 1;
total_train_time = total_loop_time*numel(data_scope);
train_percent = 0.95;

train_time = 0;
for train_loop_time = 1:total_loop_time   
    total_loss = {};
    total_learnRate = {};
    for load_scope = 1:numel(data_scope)
        clearvars -except total_loop_time train_loop_time total_train_time load_scope save_path savePath_mat savePath_txt ...
            bias_scope amp_scope_ini data_scope loop_train_num train_percent train_time total_data_num total_loss total_learnRate ...
            velocity averageGrad averageSqGrad tStart tic load_path_ini ori_rate rec_rate
        pause(10)

        %% Parameter settings
        equal_order = 30;
        headwindow = equal_order-(fix(equal_order/2)+1);
        rate_times = rec_rate/ori_rate;
        add_zero = rate_times*equal_order/2;
        pilot_length = 2047;

        split_num = 10;

        % inputSize = equal_order*rate_times;
        % inputSize = equal_order;
        inputSize = 30;
        outputSize = 1;
        hiddlayer_num = 3;
        maxEpochs = 60;
        LearnRateDropPeriod = 8;
        LearnRateDropFactor = 0.5;
        inilearningRate = 1e-2;
        velocity = [];
        momentum = 0.9;

        %% Load data
        data = split_data(amp_scope_ini,bias_scope);
        load_begin = data_scope{load_scope}(1);
        load_end = data_scope{load_scope}(2);
        data_num = load_end-load_begin+1;

        train_time = train_time+1;
        xValidation = {};
        yValidation = {};
        xTrain = {};
        yTrain = {};

        for data_loop = 1:numel(data)
            load_data_dnn
        end
        
        %% Shuffling data
        % Train_cell_num = numel(xTrain);
        % X = xTrain;
        % Y = yTrain;
        % X = cell2mat(X);
        % Y = cell2mat(Y);
        % numOber = size(X,2);
        % idx = randperm(numOber);
        % X = X(:,idx);
        % Y = Y(:,idx);
        % X = mat2cell(X,size(X,1),repmat(size(X,2)/Train_cell_num,1,Train_cell_num));
        % Y = mat2cell(Y,size(Y,1),repmat(size(Y,2)/Train_cell_num,1,Train_cell_num));
        % xTrain = X;
        % yTrain = Y;
        % 
        % Validation_cell_num = numel(xValidation);
        % X = xValidation;
        % Y = yValidation;
        % X = cell2mat(X);
        % Y = cell2mat(Y);
        % numOber = size(X,2);
        % idx = randperm(numOber);
        % X = X(:,idx);
        % Y = Y(:,idx);
        % X = mat2cell(X,size(X,1),repmat(size(X,2)/Validation_cell_num,1,Validation_cell_num));
        % Y = mat2cell(Y,size(Y,1),repmat(size(Y,2)/Validation_cell_num,1,Validation_cell_num));
        % xValidation = X;
        % yValidation = Y;
        % clear X Y idx
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   begin
        xValidation_final = cell(1,rate_times);
        yValidation_final = cell(1,rate_times);
        colNum_perRate = size(xValidation{1},2)/rate_times;
        for i = 1:numel(xValidation)
            for j = 1:rate_times
                xValidation_final{j} = [ xValidation_final{j} xValidation{i}( :,(j-1)*colNum_perRate+1:j*colNum_perRate ) ];
                yValidation_final{j} = [ yValidation_final{j} yValidation{i}( :,(j-1)*colNum_perRate+1:j*colNum_perRate ) ];
            end
        end
        for i = 1:numel(xValidation_final)
            colNum_tmp = size(xValidation_final{i},2);
            idx = randperm(colNum_tmp);
            xValidation_final{i} = xValidation_final{i}(:,idx);
            yValidation_final{i} = yValidation_final{i}(:,idx);
        end
        xValidation_final = cell2mat(xValidation_final);
        yValidation_final = cell2mat(yValidation_final);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   end
        %% Initialize network
        miniBatchSize = ceil(numel(xTrain)/80);
        numIterPerEpoch = ceil(numel(xTrain)/miniBatchSize);
        validationFrequency = floor(numIterPerEpoch/4);

        %% Build network
        dnn_option
        dlnet = dlnetwork(lgraph);

        %% Train network      
        net_path = save_path+"/net/looptime"+train_loop_time+"/net"+train_time;
        if(~exist(net_path,'dir'))
            mkdir(char(net_path));
        end
        tic
        [ dlnet, velocity, losss, learnRate_save ] = dnn_train_custom_2(rate_times,maxEpochs, xTrain, yTrain, xValidation_final, yValidation_final , ...
                                                        numIterPerEpoch, miniBatchSize, dlnet, velocity, inilearningRate, momentum,...
                                                        train_time, total_train_time, LearnRateDropPeriod, LearnRateDropFactor, validationFrequency);
        toc
        total_loss{train_time} = losss.';
        total_learnRate{train_time} = learnRate_save.';
        save(net_path+"/net.mat",'dlnet');  % Save the trained network

    end

end 