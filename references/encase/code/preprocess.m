%%%%%%%%%%%%%%%%%%%%%%%%%
% step 1: read raw data
% step 2: split qrs
% step 3: write
% 
% data format sample: 
% A000001,N,1,1,1,1,1
%%%%%%%%%%%%%%%%%%%%%%%%%


fin = fopen('../../REFERENCE.csv');
str=fgetl(fin);
fout1 = fopen('../../data1/short.csv','w');
fout2 = fopen('../../data1/long.csv','w');
fout3 = fopen('../../data1/QRSinfo.csv','w');

n_iter = 10;
ratio = 0.68;

while ischar(str)
    line=textscan(str,'%s');
    tmp = strsplit(line{1}{1}, ',');
    pid = tmp{1};
    label = tmp{2};
    
    disp(pid);
    [tm,ecg,fs,siginfo]=rdmat(strcat('../../training2017/', pid));
    [QRS,sign,en_thres] = qrs_detect2(ecg',0.25,0.6,fs);
    QRS_info = diff([0 QRS length(ecg)]);
    
    THRES = 0.6;
    iter = 0;
    while max(QRS_info) > fs*2
        iter = iter + 1;
        if iter >= n_iter
            break
        end
        THRES = ratio * THRES;
        [QRS,sign,en_thres] = qrs_detect2(ecg',0.25,THRES,fs);
        QRS_info = diff([0 QRS length(ecg)]);
    end
    if max(QRS_info) > length(ecg)*0.9
        [QRS,sign,en_thres] = qrs_detect2(ecg'*2,0.25,0.6,fs);
        QRS_info = diff([0 QRS length(ecg)]);
    end
        
        
    
    %%% write long
    tmp_len = length(ecg);
    fprintf(fout2, '%s,', pid);
    fprintf(fout2, '%s,', label);
    fprintf(fout2, '%f,',ecg(1:tmp_len-1));
    fprintf(fout2, '%f\n',ecg(tmp_len));
    
    %%% write short
    for i = 1:(length(QRS)-1)
        %%% +1 to avoid overlap
        segment = ecg(QRS(i)+1:QRS(i+1));
        tmp_len = length(segment);
        fprintf(fout1, '%s,', pid);
        fprintf(fout1, '%s,', label);
        fprintf(fout1, '%f,',segment(1:tmp_len-1));
        fprintf(fout1, '%f\n',segment(tmp_len));
    end
    
    %%% write qrs info
    % add 0 and length to head and tail, diff to get length of each
    % segment, notice that the first and the last is not accurate
    
    tmp_len = length(QRS_info);
    fprintf(fout3, '%s,', pid);
    fprintf(fout3, '%s,', label);
    if tmp_len < 2
        %%% if QRS only have one split
        fprintf(fout3, '%f\n',QRS_info(tmp_len));
    else
        fprintf(fout3, '%f,',QRS_info(1:tmp_len-1));
        fprintf(fout3, '%f\n',QRS_info(tmp_len));
    end
    
    str=fgetl(fin);
    
%     break;
end

fclose(fin);
fclose(fout1);
fclose(fout2);
fclose(fout3);

