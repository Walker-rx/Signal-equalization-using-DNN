%% Load data
if data_loop == 1
    save_amp = [];
    save_band_power = [];
    save_bias = []; 
    bias_all = [];
end
xTrain_loop = [];  % received signal
yTrain_loop = [];  % transmitted signal

test_num = 0;
data_tmp = data{data_loop};
for row_loop = 1:size(data_tmp,1)        
    amp_folder = data_tmp(row_loop,1);
    bias_loop_data = data_tmp(row_loop,2:end);
    bias_loop_data(find(bias_loop_data==0)) = [];
    bias_all = [bias_all bias_loop_data];
    for bias_loop = 1:length(bias_loop_data)
        bias_folder = bias_loop_data(bias_loop);
        test_num = test_num + 1;
        load_path = load_path_ini + "/data/"+ori_rate/1e6+"M/amp"+amp_folder+"/bias"+bias_folder+"/mat";
        fprintf(" %d looptimes , %d training times , load amp = %f , bias = %d , \n load begin = %d , load end = %d \n",...
            train_loop_time,train_time,amp_folder,bias_folder,load_begin,load_end);
        load_data_pilot
        totalNum = data_num;
        trainNum = floor(totalNum*train_percent);
        xTrain_tmp = pilot_received_load(1:trainNum);
        yTrain_tmp = pilot_ori_load(1:trainNum);
        xTest_tmp = pilot_received_load(trainNum+1:end);
        yTest_tmp = pilot_ori_load(trainNum+1:end);

        amp_loop = 32000*amp_folder;
        save_amp = [ save_amp amp_folder ];
        save_bias = [ save_bias bias_folder];
        yTrain_tmp = cellfun(@(cell1)(cell1*amp_loop),yTrain_tmp,'UniformOutput',false);
        yTest_tmp = cellfun(@(cell1)(cell1*amp_loop),yTest_tmp,'UniformOutput',false);

        xTrain_loop = [xTrain_loop xTrain_tmp];
        yTrain_loop = [yTrain_loop yTrain_tmp];

        xTest_name = ['xTest',num2str(test_num)];
        yTest_name = ['yTest',num2str(test_num)];
        eval([xTest_name,'=xTest_tmp;']);
        eval([yTest_name,'=yTest_tmp;']);

        clear x y
    end
end

%%  Normalize data
test_num = round(test_num);
load_norm_path = "/home/oem/Users/ruoxu/channel-estimation-using-DNN/data_save/norm_factor/";
norm_mat = load(load_norm_path+"/save_norm.mat");
norm_names = fieldnames(norm_mat);
norm_factor = gather(eval(strcat('norm_mat.',norm_names{1})));

yTrain_loop = cellfun(@(cell1)(cell1*norm_factor),yTrain_loop,'UniformOutput',false);

for i = 1:test_num
    save_band_power = [save_band_power bandpower(yTrain_loop{2+(i-1)*trainNum})];
end

for i = 1:test_num
    yTest_nor = eval(['yTest',num2str(i)]);
    yTest_nor = cellfun(@(cell1)(cell1*norm_factor),yTest_nor,'UniformOutput',false);
    eval([['yTest',num2str(i)],'= yTest_nor;']);
end

%%  Reshape data
for i = 1:numel(xTrain_loop)
    x_rate = [];
    for j = 1:rate_times
        x_rate_loop = xTrain_loop{i}(j:rate_times:end);
        x_rate_loop = toeplitz(x_rate_loop(equal_order:-1:1),x_rate_loop(equal_order:end));
        x_rate = [x_rate x_rate_loop];
    end
    xTrain_loop{i} = x_rate;
%     xTrain_loop{i} = [xTrain_loop{i}; single( bias_loop_data(floor((i-1)/trainNum)+1) )*ones(1,size(xTrain_loop{i},2) )];
    yTrain_loop{i} = reshape(yTrain_loop{i},outputSize,[]);
    % yTrain_loop{i} = yTrain_loop{i}(:,1:size(xTrain_loop{i},2));
    yTrain_loop{i} = yTrain_loop{i}(:,1:size(xTrain_loop{i},2)/rate_times);
    yTrain_loop_tmp = yTrain_loop{i};
    yTrain_loop{i} = [];
    for k = 1:rate_times
        yTrain_loop{i} = [yTrain_loop{i} yTrain_loop_tmp];
    end
end

for i = 1:test_num
    xtop_tem = eval(['xTest',num2str(i)]);
    ytop_tem = eval(['yTest',num2str(i)]);
    for j = 1:numel(xtop_tem)
        x_rate = [];
        for k = 1:rate_times
            x_rate_loop = xtop_tem{j}(k:rate_times:end);
            x_rate_loop = toeplitz(x_rate_loop(equal_order:-1:1),x_rate_loop(equal_order:end));
            x_rate = [x_rate x_rate_loop];
        end
        xtop_tem{j} = x_rate;
%         xtop_tem{j} = [xtop_tem{j}; single( bias_loop_data(i) )*ones(1,size(xtop_tem{j},2) )];
        ytop_tem{j} = reshape(ytop_tem{j},outputSize,[]);
        % ytop_tem{j} = ytop_tem{j}(:,1:size(xtop_tem{j},2));
        ytop_tem{j} = ytop_tem{j}(:,1:size(xtop_tem{j},2)/rate_times);
        ytop_tem_2 = ytop_tem{j};
        ytop_tem{j} = [];
        for r = 1:rate_times
            ytop_tem{j} = [ytop_tem{j} ytop_tem_2];
        end
    end
    eval([['xTest',num2str(i)],'= xtop_tem;']);
    eval([['yTest',num2str(i)],'= ytop_tem;']);
end

for i = 1:test_num
    xValidation_tem = eval(['xTest',num2str(i)]);
    yValidation_tem = eval(['yTest',num2str(i)]);
    if numel(xValidation_tem{1})>1
        xValidation = [xValidation,xValidation_tem{1},xValidation_tem{2}];
        yValidation = [yValidation,yValidation_tem{1},yValidation_tem{2}];
    else
        xValidation = [xValidation,xValidation_tem{1}];
        yValidation = [yValidation,yValidation_tem{1}];
    end
end
xTrain = [xTrain xTrain_loop];
yTrain = [yTrain yTrain_loop];

%%
