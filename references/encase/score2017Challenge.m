% This script will score your algorithm for classification accuracy, based on the reference classification results.
% Your final score for the challenge will be evaluated on the whole hidden test set.
%
% This script requires that you first run generateValidationSet.m
%
%
% Written by: Chengyu Liu and Qiao Li January 20 2017
%             chengyu.liu@emory.edu  qiao.li@emory.edu
%
% Last modified by:
%
%

clear all;

%% Load the answer classification results
fid = fopen('answers.txt','r');
if(fid ~= -1)
    ANSWERS = textscan(fid, '%s %s','Delimiter',',');
else
    error('Could not open users answer.txt for scoring. Run the generateValidationSet.m script and try again.')
end
fclose(fid);

%% Load the reference classification results
reffile = ['validation' filesep 'REFERENCE.csv'];
fid = fopen(reffile, 'r');
if(fid ~= -1)
    Ref = textscan(fid,'%s %s','Delimiter',',');
else
    error(['Could not open ' reffile ' for scoring. Exiting...'])
end
fclose(fid);

RECORDS = Ref{1};
target  = Ref{2};
N       = length(RECORDS);

a = find(strcmp(ANSWERS{2},'N'));
b = find(strcmp(ANSWERS{2},'A'));
c = find(strcmp(ANSWERS{2},'O'));
d = find(strcmp(ANSWERS{2},'~'));
ln = length(a)+length(b)+length(c)+length(d);
if(length(ANSWERS{2}) ~= ln);
    error('Input must contain only N, A, O or ~');
end

%% Scoring
% We do not assume that the references and the answers are sorted in the
% same order, so we search for the location of the individual records in answer.txt file.
AA=zeros(4,4);

for n = 1:N
    rec = RECORDS{n};
    i = strmatch(rec, ANSWERS{1});
    if(isempty(i))
        warning(['Could not find answer for record ' rec '; treating it as NOISE (~).']);
        this_answer = '~';
    else
        this_answer = ANSWERS{2}(i);
    end
    switch target{n}
        case 'N'
            if strcmp(this_answer,'N')
                AA(1,1) = AA(1,1)+1;
            elseif strcmp(this_answer,'A')
                AA(1,2) = AA(1,2)+1;
            elseif strcmp(this_answer,'O')
                AA(1,3) = AA(1,3)+1;
            elseif strcmp(this_answer,'~')
                AA(1,4) = AA(1,4)+1;
            end
        case 'A'
            if strcmp(this_answer,'N')
                AA(2,1) = AA(2,1)+1;
            elseif strcmp(this_answer,'A')
                AA(2,2) = AA(2,2)+1;
            elseif strcmp(this_answer,'O')
                AA(2,3) = AA(2,3)+1;
            elseif strcmp(this_answer,'~')
                AA(2,4) = AA(2,4)+1;
            end
        case 'O'
            if strcmp(this_answer,'N')
                AA(3,1) = AA(3,1)+1;
            elseif strcmp(this_answer,'A')
                AA(3,2) = AA(3,2)+1;
            elseif strcmp(this_answer,'O')
                AA(3,3) = AA(3,3)+1;
            elseif strcmp(this_answer,'~')
                AA(3,4) = AA(3,4)+1;
            end
        case '~'
            if strcmp(this_answer,'N')
                AA(4,1) = AA(4,1)+1;
            elseif strcmp(this_answer,'A')
                AA(4,2) = AA(4,2)+1;
            elseif strcmp(this_answer,'O')
                AA(4,3) = AA(4,3)+1;
            elseif strcmp(this_answer,'~')
                AA(4,4) = AA(4,4)+1;
            end
    end
end

F1n=2*AA(1,1)/(sum(AA(1,:))+sum(AA(:,1)));
F1a=2*AA(2,2)/(sum(AA(2,:))+sum(AA(:,2)));
F1o=2*AA(3,3)/(sum(AA(3,:))+sum(AA(:,3)));
F1p=2*AA(4,4)/(sum(AA(4,:))+sum(AA(:,4)));
F1=(F1n+F1a+F1o)/3;


str = ['F1 measure for Normal rhythm:  ' '%1.4f\n'];
fprintf(str,F1n)
str = ['F1 measure for AF rhythm:  ' '%1.4f\n'];
fprintf(str,F1a)
str = ['F1 measure for Other rhythm:  ' '%1.4f\n'];
fprintf(str,F1o)
str = ['F1 measure for Noisy recordings:  ' '%1.4f\n'];
fprintf(str,F1p)
str = ['Final F1 measure:  ' '%1.4f\n'];
fprintf(str,F1)



