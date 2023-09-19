clear
close all

pilot_length = 2047;
zero_length = 3000;
data_length = 10000;
ori_rate = 30e6;
rec_rate = 60e6;
times = rec_rate/ori_rate;
equal_order = 30;
M = 2;

bias_scope = 0.05:0.04:0.85;
amp_scope = [0.005 0.007 0.015 0.024 0.034 0.045 0.08 0.18 0.25 0.3 0.48082 0.64058 0.8003 1];


load_begin = 1;
load_end = 60;
load_num = load_end-load_begin+1;
load_path_ori = "data_save/light_data_6.18";
% load_path_ori = "/home/xliangseu/ruoxu/channel-estimation-using-DNN/data_save/light_data_4.14/";

ver = 1;
for bias_loop = 1:length(bias_scope)
    bias = bias_scope(bias_loop);
    save_path = "data_save/light_data_6.18/result1/cal_nmse/v"+ver+"/bias"+bias;
    if(~exist(save_path,'dir'))
        mkdir(char(save_path));
    end
    for amp_loop = 1:length(amp_scope)       
        nmse_all = 0;
        errornum_all = 0;
        length_all = 0;
        ps_all = 0;
        pn_all = 0;

        amp = amp_scope(amp_loop);
        load_path = load_path_ori + "/data/"+ori_rate/1e6+"M/amp"+amp+"/bias"+bias+"/mat";
        load_mmse

        for load_loop = load_begin:load_end
            signal_ori = signal_ori_all{load_loop};
            signal_received = signal_received_all{load_loop};
            fin_syn_point = fin_syn_point_all{load_loop};
            [nmseNum,length_loop,ps_loop,pn_loop,errornum_ls_loop,error_location,data_demod_ori,data_demod_hat] ...
            = calculate_nmse(signal_ori,signal_received,pilot_length,zero_length,data_length,fin_syn_point,times,equal_order,M);
            
            nmse_all = nmse_all+nmseNum;
            errornum_all = errornum_all+errornum_ls_loop;
            length_all = length_all+length_loop;
            ps_all = ps_all+ps_loop;
            pn_all = pn_all+pn_loop;

            snr_loop = 10*log10(ps_loop/pn_loop);
            ser_loop = errornum_ls_loop/length_loop;
            fprintf("bias = %f , amp = %f , snr = %f , ser = %.6g , nmse = %f \r\n",bias,amp,snr_loop,ser_loop,nmseNum);
        end
        nmse_mean = nmse_all/load_num;
        snr_mean = 10*log10(ps_all/pn_all);
        ser_mean = errornum_all/length_all;
        
        if amp_loop == 1
            famp = fopen(save_path+"/amp.txt",'w');
            fsnr = fopen(save_path+"/snr.txt",'w');
            fser = fopen(save_path+"/ser.txt",'w');
            fnmse = fopen(save_path+"/nmse.txt",'w');
            fprintf(famp,'amp = \r\n');
            fprintf(fsnr,'snr = \r\n');
            fprintf(fser,'ser = \r\n');
            fprintf(fnmse,'mmse = \r\n');
        else
            famp = fopen(save_path+"/amp.txt",'a');
            fsnr = fopen(save_path+"/snr.txt",'a');
            fser = fopen(save_path+"/ser.txt",'a');
            fnmse = fopen(save_path+"/nmse.txt",'a');
        end
        fprintf(famp,' %f \r\n',amp);
        fprintf(fsnr,' %f \r\n',snr_mean);
        fprintf(fser,' %.6g \r\n',ser_mean);
        fprintf(fnmse,' %f \r\n',nmse_mean);
        fclose(famp);
        fclose(fsnr);
        fclose(fser);
        fclose(fnmse);       
    end
end



    