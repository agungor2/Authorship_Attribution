%%
%Top 100 words ttest2 analysis one vs others
clear all
load('Vocabulary_wstopwords.mat')
load('ml_challenge_data_wstopwords.mat')
most_common = csvread('most_common_index.csv');
word2vec = csvread('word2vec_data_stop_words.csv');

%Find the less occurent words:
a = unique(txt_pieces);
out = [histc(txt_pieces(:),a),a];
less_occurent = sortrows(out,1,'ascend');
%most_common = less_occurent(1:100,:);
unique_author = unique(aid);
all_ttest_result = zeros(length(most_common),length(unique_author));

for i=1:length(most_common)
    tic
   all_dist=cell(length(unique_author),1);
    tic
   for j=1:length(unique_author)
       %Choose all elements and make it a chain
       tmp_txt = txt_pieces(aid==unique_author(j),:);
       tmp_txt = tmp_txt';
       tmp_txt = tmp_txt(:);
       %Takeout the zero element
       tmp_txt=tmp_txt(tmp_txt~=0);
       index = find(tmp_txt==most_common(i,2));
       %takeout the first and last element indices
       index(find(index == 1 | index == length(tmp_txt)))=[];
       at1 = word2vec(tmp_txt(index -1),:);
       at2 = word2vec(tmp_txt(index), :);
       at3= word2vec(tmp_txt(index + 1),:);
       dist = zeros(size(at1,1),1);
       %Euclidian distance for the vectors
       for k = 1:size(at1,1)
           %warning('off','all')
          %dist(k) = pdist([at1(k,:);at3(k,:)],'cosine')/(pdist([at1(k,:);at2(k,:)],'cosine') + pdist([at2(k,:);at3(k,:)],'cosine')); 
          dist(k) = dist_cosine(at1(k,:),at3(k,:))/(dist_cosine(at1(k,:),at2(k,:)) + dist_cosine(at2(k,:),at3(k,:)));
       end
   
       all_dist{j} = dist;
   end
   toc
   tic
   %Now we can do ttest2 analysis for each element
   ttest_result = zeros(length(unique_author),1);
   for l = 1:length(unique_author)
       x = all_dist{l};
       y=[];
       for m =1:length(unique_author)
           if m==l
               y_temp = [];
           else
               y_temp = all_dist{m};
           end
           y_new = vertcat(y,y_temp);  
           y = y_new;
       end
           [h,p,ci,stats] = ttest2(x,y);
           ttest_result(l) = p;
   end
   toc
      ttest_result
      all_ttest_result(i,:) = ttest_result;
      
      toc
end

%Find the rows that are below than 0.05
tmp = all_ttest_result(1:20,:);
tmp1 = zeros(size(tmp,1),length(unique_author));
for i =1:size(tmp,1)
for j = 1:length(unique_author)
if tmp(i,j)<0.005 && tmp(i,j)>0
tmp1(i,j)=1;
end
end
end

%Plot them as black and white
[r, c] = size(tmp1);                          % Get the matrix size
imagesc((1:c)+0.5, (1:r)+0.5, tmp1);          % Plot the image
colormap(gray);                              % Use a gray colormap
axis equal                                   % Make axes grid sizes equal
set(gca, 'XTick', 1:(c+1), 'YTick', 1:(r+1), ...  % Change some axes properties
         'XLim', [1 c+1], 'YLim', [1 r+1], ...
         'GridLineStyle', '-', 'XGrid', 'on', 'YGrid', 'on');
title('one vs others for most occurent 20 words')
xlabel('Author1-50') % x-axis label
ylabel('Word1-20') % y-axis label


%%