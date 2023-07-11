signal_ori_mat = load(load_path+"/save_signal_ori.mat");
signal_received_mat = load(load_path+"/save_signal_received_real_send1.mat");
fin_mat = load(load_path+"/save_fin_syn_point_real_send1.mat");
upsample_norm_mat = load(load_path+"/save_upsample_norm.mat");

signal_ori_names = fieldnames(signal_ori_mat);
signal_received_names = fieldnames(signal_received_mat);
fin_names = fieldnames(fin_mat);
upsample_norm_names = fieldnames(upsample_norm_mat);


signal_ori_all = cell(1,load_end-load_begin+1);
signal_received_all = cell(1,load_end-load_begin+1);
fin_syn_point_all = cell(1,load_end-load_begin+1);
upsample_norm_all = cell(1,load_end-load_begin+1);


for name_order = load_begin:load_end

    signal_ori = gather(eval(strcat('signal_ori_mat.',signal_ori_names{name_order})));
    signal_received = gather(eval(strcat('signal_received_mat.',signal_received_names{name_order})));
    fin_syn_point = gather(eval(strcat('fin_mat.',fin_names{name_order})));
    upsample_norm = gather(eval(strcat('upsample_norm_mat.',upsample_norm_names{name_order})));

    signal_ori_all{name_order} = signal_ori;
    signal_received_all{name_order} = signal_received;
    fin_syn_point_all{name_order} = fin_syn_point;
    upsample_norm_all{name_order} = upsample_norm;
            
end

clear signal_ori_mat signal_received_mat fin_mat upsample_norm_mat signal_ori_names signal_received_names fin_names upsample_norm_names

