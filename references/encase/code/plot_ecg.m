
fin = fopen('../../data/REFERENCE.csv');
str=fgetl(fin);
cnt = 0;

while ischar(str)
    cnt = cnt + 1;
    line=textscan(str,'%s');
    tmp = strsplit(line{1}{1}, ',');
    pid = tmp{1};
    label = tmp{2};
    
    if cnt < 6352
        str=fgetl(fin);
        continue;
    end
    
    disp(pid);
    [tm,ecg,fs,siginfo]=rdmat(strcat('../../training2017/', pid));
    [QRS,sign,en_thres] = qrs_detect2(ecg',0.25,0.6,fs);
    
    fig = figure();
    fig.PaperPosition = [0 0 30 9];
    set(fig, 'Visible', 'off');
    
    plot(ecg);
    max_num = max(ecg);
    min_num = min(ecg);
    hold on;
    for i = 1:length(QRS)
        plot([QRS(i) QRS(i)], [min_num max_num], 'Color', [1 0.5 0.5], 'LineStyle', ':');
    end
    
    my_title = strcat(pid,'\_', label, '\_', num2str(length(QRS)+1));
    my_path = strcat('../../img/img1/', pid,'_', label, '_', num2str(length(QRS)+1));
    title(my_title);
    saveas(fig, my_path, 'png');

    str=fgetl(fin);
    
%     break;
end

fclose(fin);

