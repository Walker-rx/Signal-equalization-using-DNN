function [nmseNum,length_loop,ps_loop,pn_loop,errornum_ls_loop,error_location,data_demod_ori,data_demod_hat] ...
    = calculate_mmse(signal_ori,signal_received,pilot_length,zero_length,data_length,fin_syn_point,times,equal_order,M)
                                             
    signal_downsample = signal_received(fin_syn_point:times:end);
    noise = signal_downsample(pilot_length+100:pilot_length+100+zero_length/2-1);
    pn_loop = bandpower(noise);
    if (pilot_length+zero_length+1+data_length-1) <= length(signal_downsample)
        data_received = signal_downsample(pilot_length+zero_length+1:pilot_length+zero_length+1+data_length-1);
    else
        data_received = signal_downsample(pilot_length+zero_length+1:end);
    end
    p = bandpower(data_received);
    ps_loop = p - pn_loop;
    
    signal_hat = signal_equal_ls(signal_ori,signal_received,times,fin_syn_point,pilot_length,equal_order);
    data_hat = signal_hat(pilot_length+zero_length+1:end);
    data_demod_hat = pamdemod(data_hat,M);
    
    data_ori = signal_ori(pilot_length+zero_length+1:end);
    data_demod_ori = pamdemod(data_ori,M);

    if length(data_hat) < length(data_ori)
        data_ori = data_ori(1:length(data_hat));
        compare_length = length(data_hat);
        errornum_ls_loop = sum(data_demod_hat ~= data_demod_ori(1:compare_length));
        error_location = find(data_demod_hat ~= data_demod_ori(1:compare_length));
    else
        data_hat = data_hat(1:length(data_ori));
        compare_length = data_length;
        errornum_ls_loop = sum(data_demod_hat(1:compare_length) ~= data_demod_ori);
        error_location = find(data_demod_hat(1:compare_length) ~= data_demod_ori);
    end
    length_loop = compare_length;

    nmseNum_fun = @(hat,exp)10*log10(sum(sum((hat-exp).^2))/sum(sum(exp.^2)));
    nmseNum = nmseNum_fun(data_hat,data_ori);
                
end