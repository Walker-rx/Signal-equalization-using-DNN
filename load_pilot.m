signal_ori_mat = load(load_path+"/save_signal_ori.mat");
signal_received_mat = load(load_path+"/save_signal_received_real_send1.mat");
fin_mat = load(load_path+"/save_fin_syn_point_real_send1.mat");
upsample_norm_mat = load(load_path+"/save_upsample_norm.mat");

signal_ori_names = fieldnames(signal_ori_mat);
signal_received_names = fieldnames(signal_received_mat);
fin_names = fieldnames(fin_mat);
upsample_norm_names = fieldnames(upsample_norm_mat);

pilot_ori_load = cell(1,data_num);
pilot_received_load = cell(1,data_num);
upsample_norm = zeros(1,data_num);

load_data_loop = 0;
for name_order = load_begin:load_end
    load_data_loop = load_data_loop+1;
    signal_ori = gather(eval(strcat('signal_ori_mat.',signal_ori_names{name_order})));
    signal_received = gather(eval(strcat('signal_received_mat.',signal_received_names{name_order})));
    fin_syn_point = gather(eval(strcat('fin_mat.',fin_names{name_order})));
    upsample_norm(load_data_loop) = gather(eval(strcat('upsample_norm_mat.',upsample_norm_names{name_order})));
    pilot_ori = signal_ori(1:pilot_length);
    if headwindow*rate_times > fin_syn_point-1
        signal_addzero = [ zeros(1,headwindow*rate_times-(fin_syn_point-1)) , signal_received ];
    else
        start_point = fin_syn_point - headwindow*rate_times;
        signal_addzero = signal_received(start_point:end);
    end
    pilot_received = signal_addzero(1:pilot_length*rate_times);    
    pilot_ori_load{load_data_loop} = pilot_ori;
    pilot_ori_load{load_data_loop} =  pilot_ori_load{load_data_loop}/upsample_norm(load_data_loop);
    pilot_received_load{load_data_loop} = pilot_received;
end
pilot_ori_load = gather(pilot_ori_load);
pilot_received_load = gather(pilot_received_load);
clear signal_ori_mat signal_received_mat fin_mat signal_ori_names signal_received_names fin_names upsample_norm_mat upsample_norm_names

