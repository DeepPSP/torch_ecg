function preprocess_sub(recordName)
fout1 = fopen('data1/short.csv','w');
fout2 = fopen('data1/long.csv','w');
fout3 = fopen('data1/QRSinfo.csv','w');
pid = recordName;
label = 1;
%label is not used

[tm,ecg,fs,siginfo]=rdmat(recordName);
[QRS,sign,en_thres] = qrs_detect2(ecg',0.25,0.6,fs);
QRS_info = diff([0 QRS length(ecg)]);

n_iter = 10;
ratio = 0.68;
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
    
%%% long数据在ecg中
tmp_len = length(ecg);
fprintf(fout2, '%s,', pid);
fprintf(fout2, '%s,', label);
fprintf(fout2, '%f,',ecg(1:tmp_len-1));
fprintf(fout2, '%f\n',ecg(tmp_len));

%%% short数据在segment中
for i = 1:(length(QRS)-1)
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
    fprintf(fout3, '%f\n',QRS_info(tmp_len));
else
    fprintf(fout3, '%f,',QRS_info(1:tmp_len-1));
    fprintf(fout3, '%f\n',QRS_info(tmp_len));
end

%向answers.txt写文件名
fout4 = fopen('answers.txt','a');
fprintf(fout4, '%s,', recordName);
fclose(fout4);

fclose(fout1);
fclose(fout2);
fclose(fout3);

