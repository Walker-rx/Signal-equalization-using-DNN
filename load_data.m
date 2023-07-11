signal_ori_mat = load(load_path+"/save_signal_ori.mat");
signal_received_mat = load(load_path+"/save_signal_received_real_send1.mat");
fin_mat = load(load_path+"/save_fin_syn_point_real_send1.mat");
upsample_norm_mat = load(load_path+"/save_upsample_norm.mat");

signal_ori_names = fieldnames(signal_ori_mat);
signal_received_names = fieldnames(signal_received_mat);
fin_names = fieldnames(fin_mat);
upsample_norm_names = fieldnames(upsample_norm_mat);

data_ori_load = cell(1,data_num*split_num);
data_received_load = cell(1,data_num*split_num);
upsample_norm = zeros(1,data_num);
split_length = data_length/split_num;
% x = cell(data_num,1);
% y = cell(data_num,1);
load_data_loop = 0;
for name_order = load_begin:load_end
    load_data_loop = load_data_loop+1;
    signal_ori = gather(eval(strcat('signal_ori_mat.',signal_ori_names{name_order})));
    signal_received = gather(eval(strcat('signal_received_mat.',signal_received_names{name_order})));
    fin_syn_point = gather(eval(strcat('fin_mat.',fin_names{name_order})));
    upsample_norm(load_data_loop) = gather(eval(strcat('upsample_norm_mat.',upsample_norm_names{name_order})));

    data_ori = signal_ori(pilot_length+zero_length+1:end);
    data_received = signal_received(fin_syn_point + (pilot_length+zero_length)*rate_times : end);

    for i_for_load = 1:split_num
        data_ori_load{split_num*(load_data_loop-1)+i_for_load} = data_ori(split_length*(i_for_load-1)+1 : split_length*i_for_load);
        data_ori_load{split_num*(load_data_loop-1)+i_for_load} =  data_ori_load{split_num*(load_data_loop-1)+i_for_load}/upsample_norm(load_data_loop);
        data_received_load{split_num*(load_data_loop-1)+i_for_load} = [ zeros(1,headwindow*rate_times) data_received(split_length*rate_times*(i_for_load-1)+1 : split_length*rate_times*i_for_load) ];
    end
end
data_ori_load = gather(data_ori_load);
data_received_load = gather(data_received_load);
clear signal_ori_mat signal_received_mat fin_mat signal_ori_names signal_received_names fin_names upsample_norm_mat upsample_norm_names

