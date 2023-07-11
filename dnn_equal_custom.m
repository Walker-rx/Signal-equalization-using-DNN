clear
close all
tStart = tic;

ver = 3;

t = datetime('now');
folder = '6.18';
ori_rate = 30e6;
rec_rate = 60e6;
% load_path_ini = "/home/xliangseu/ruoxu/equalization-using-DNN/data_save/light_data_"+folder;
load_path_ini = "/home/oem/Users/ruoxu/equalization-using-DNN/data_save/light_data_"+folder;
save_path_ini = "data_save/light_data_"+folder;
save_path = save_path_ini + "/result1/"+t.Month+"."+t.Day+"/mix_bias_amp/Threenonlinear"+ver;   

if(~exist(save_path,'dir'))
    mkdir(char(save_path));
end

%% Network parameters
bias_scope = 0.05:0.04:0.85;
amp_scope_ini = [0.005 0.007 0.015 0.024 0.034 0.045 0.08 0.18 0.25 0.3 0.48082 0.64058 0.8003 1];
% bias_scope = 0.45;
% amp_scope_ini = 0.18;

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
total_train_time = total_loop_time*loop_train_num;
train_percent = 0.95;

train_time = 0;
for train_loop_time = 1:total_loop_time   
    total_loss = {};
    total_learnRate = {};
    for load_scope = 1:numel(data_scope)
        clearvars -except total_loop_time train_loop_time total_train_time load_scope save_path savePath_mat savePath_txt ...
            bias_scope amp_scope_ini data_scope loop_train_num train_percent train_time total_data_num total_loss total_learnRate ...
            velocity averageGrad averageSqGrad tStart tic load_path_ini ori_rate rec_rate dlnet
        pause(10)

        %% Parameter settings
        equal_order = 30;
        headwindow = equal_order-(fix(equal_order/2)+1);
        rate_times = rec_rate/ori_rate;
        % add_zero = rate_times*equal_order/2;
        pilot_length = 2047;
        zero_length = 3000;
        data_length = 10000;
        split_num = 10;

        inputSize = equal_order+1;
        outputSize = 1;
        hiddlayer_num = 4;
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
            load_dnn
        end
        
        %% Shuffling data
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

        %% Initialize network
        miniBatchSize = ceil(numel(xTrain)/80);
        numIterPerEpoch = ceil(numel(xTrain)/miniBatchSize);
        validationFrequency = floor(numIterPerEpoch/4);

        %% Build network
        dnn_option
        if ~exist("dlnet")
            dlnet = dlnetwork(lgraph);
        end
        
        %% Train network      
        net_path = save_path+"/net/looptime"+train_loop_time+"/net"+train_time;
        if(~exist(net_path,'dir'))
            mkdir(char(net_path));
        end
        tic
        [ dlnet, velocity, losss, learnRate_save ] = dnn_train_custom(rate_times,maxEpochs, xTrain, yTrain, xValidation_final, yValidation_final , ...
                                                        numIterPerEpoch, miniBatchSize, dlnet, velocity, inilearningRate, momentum,...
                                                        train_time, total_train_time, LearnRateDropPeriod, LearnRateDropFactor, validationFrequency);
        toc
        total_loss{train_time} = losss.';
        total_learnRate{train_time} = learnRate_save.';
        save(net_path+"/net.mat",'dlnet');  % Save the trained network

        %% Save data
        for i = 1:length(save_amp)
            if i == 1
                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'w');
            else
                save_amp_bias_txt = fopen(net_path+"/save_amp.txt",'a');
            end
            fprintf(save_amp_bias_txt," amp = %f , bias = %f ,bandpower = %f \n" , save_amp(i), save_bias(i), save_band_power(i));
            if i == length(save_amp)
                fprintf(save_amp_bias_txt," data load begin = %d , load end = %d  \n" , load_begin,load_end);
            end
            fclose(save_amp_bias_txt);
        end

        if train_time == loop_train_num
            save(save_path+"/net/looptime"+train_loop_time+"/loss.mat","total_loss");
            save(save_path+"/net/looptime"+train_loop_time+"/learnRate.mat","total_learnRate");
        end

    end

end 

%% Save parameter
save_parameter = fopen(save_path+"/save_parameter.txt",'w');
fprintf(save_parameter,"\n \n");
fprintf(save_parameter," Custom_2 training , using pilot \n");
fprintf(save_parameter," Threenonlinear ,\r\n %d iteration per epoch , \r\n ",numIterPerEpoch);
fprintf(save_parameter,"ini learningRate = %e ,\r\n DropPeriod = %d , DropFactor = %f ,\r\n ",inilearningRate,LearnRateDropPeriod, LearnRateDropFactor);
fprintf(save_parameter,"amp =");
for i = 1:length(amp_scope_ini)
    fprintf(save_parameter," %f,",amp_scope_ini(i));
end
fprintf(save_parameter,"\r\n");
fprintf(save_parameter," bias =");
for i = 1:length(bias_scope)
    fprintf(save_parameter," %f,",bias_scope(i));
end
fprintf(save_parameter,"\r\n");
% fprintf(save_parameter," data num = %d , split num = %d , train num = %d\r\n",total_cell,split_num,total_cell*split_num*train_percent);
fprintf(save_parameter," data num = %d , no split , train num = %d\r\n",total_data_num,total_data_num*train_percent);
fprintf(save_parameter," validationFrequency is floor(numIterPerEpoch/4) \n");
fprintf(save_parameter," origin rate = %e , receive rate = %e \n",ori_rate,rec_rate);
fprintf(save_parameter," Equal order = %d \n",equal_order);
fprintf(save_parameter," Hidden Units = %d \n",numHiddenUnits);
% fprintf(save_parameter," Add zero num = %d(equal) \n",add_zero);
fclose(save_parameter);

%%
fprintf("\n Training end ..." + ... 
    "\n Threenonlinear , Train cell num = %d \n",...
    total_data_num);
fprintf(" result saved in %s \n",save_path);

tEnd = toc(tStart);
disp("Total using "+floor(tEnd/60)+"min "+mod(tEnd,60)+"s")
