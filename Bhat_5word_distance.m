%%
%Top 1000 words Bhattacharyya distance
clear all
load('Vocabulary_wstopwords.mat')
load('ml_challenge_data_wstopwords.mat')
word2vec = csvread('word2vec_data_stop_words.csv');

%Find the less occurent words:
a = unique(txt_pieces);
out = [histc(txt_pieces(:),a),a];
%Delete the zero number element
out(1,:)=[];
most_common = sort(out,1,'descend');
%Take a look at the number of text versus overall authors
unique_author = unique(aid);
aid_number =[];
for i=1:length(unique_author)
    tmp=[unique_author(i),length(txt_pieces(aid==unique_author(i)))];
    aid_number = [aid_number; tmp];
end
aid_number = sort(aid_number,2,'descend');
%%We identify authors [39;45;31;21;37]
unique_author = [39;45;31;21;37];

%Choose 1000 words that have word2vec values
existing_words_index = [];
word_number = 505;

for i=1:word_number
   if word2vec(most_common(i,2))~=ones(1,300)
      existing_words_index = [existing_words_index; most_common(i,2)];
   end
end


mean_all = zeros(length(existing_words_index),length(unique_author));
variance_all = zeros(length(existing_words_index),length(unique_author));
bhat_all_dist=cell(length(existing_words_index),1);

%Split into training and testing part
all_txt =[];
all_aid =[];
for i=1:length(unique_author)
    tmp1 = txt_pieces(aid==unique_author(i),:);
    all_txt = [all_txt; tmp1];
    tmp2 = ones(size(tmp1,1),1)*unique_author(i);
    all_aid = [all_aid; tmp2];
end
%Split it into training and test data
[train_ind, val_ind, test_ind] = dividerand(size(all_txt,1),0.6,0.0,0.4);
train_txt = all_txt(train_ind,:);
test_txt = all_txt(test_ind,:);

for i=1:length(existing_words_index)
    tic
    for j =1:length(unique_author)
       %Choose all elements and make it a chain
       tmp_txt = train_txt(all_aid(train_ind)==unique_author(j),:);
       tmp_txt = tmp_txt';
       tmp_txt = tmp_txt(:);
       %Takeout the zero element
       tmp_txt=tmp_txt(tmp_txt~=0);
       index = find(tmp_txt==existing_words_index(i));
       %takeout the first, 2nd, last-1 and last element indices
       index(find(index == 1 | index == 2 | index == length(tmp_txt)-1 | index == length(tmp_txt)))=[];
       at1 = word2vec(tmp_txt(index -2),:);
       at2 = word2vec(tmp_txt(index -1),:);
       at3 = word2vec(tmp_txt(index), :);
       at4 = word2vec(tmp_txt(index + 1),:);
       at5 = word2vec(tmp_txt(index + 2),:);
       %Takeout the indices that have word2vec's as ones
       ind1 = find(~ismember(at1, ones(1,300), 'rows'));
       ind2 = find(~ismember(at2, ones(1,300), 'rows'));
       ind4 = find(~ismember(at4, ones(1,300), 'rows'));
       ind5 = find(~ismember(at5, ones(1,300), 'rows'));
       ind_final = intersect(intersect(ind1,ind2),intersect(ind4,ind5));
       at1 = at1(ind_final,:);
       at2 = at2(ind_final,:);
       at3 = at3(ind_final,:);
       at4 = at4(ind_final,:);
       at5 = at5(ind_final,:);
       
       %Cosine distance for the vectors
       dist = (dist_cosine(at1,at3) + dist_cosine(at3,at5))./(dist_cosine(at1,at2) + dist_cosine(at2,at3) +dist_cosine(at3,at4) + dist_cosine(at4,at5));
       dist = dist((~isnan(dist)));
       mean_all(i,j) = mean(dist); 
       variance_all(i,j) = var(dist);
    end
    
   %Now we can do Bhatasharya distance
   bhat_distance = zeros(length(unique_author),length(unique_author));
   for l = 1:length(unique_author)
       for m =1:length(unique_author)
           mean1 = mean_all(i,l);
           mean2 = mean_all(i,m);
           sigma1 = variance_all(i,l);
           sigma2 = variance_all(i,m); 
           tmp = Bhat_dist(sigma1,sigma2,mean1,mean2);
           bhat_distance(l,m)=tmp;
           bhat_distance(m,l)=tmp;
       end
   end
   bhat_all_dist{i} = bhat_distance;
   toc
end

at = zeros(length(existing_words_index),10);

for i=1:length(existing_words_index)
tmp = bhat_all_dist{i,1};
c=1;
    for j=1:4
        for l=j+1:5
                at(i,c)=tmp(j,l);
                c=c+1;
        end
    end
end

%Take the indices that are greater than the mean
tmp_mean = mean(at);
indices = [];
for i=1:length(mean(at))
tmp = find(at(:,i) > tmp_mean(i));
indices = [indices; tmp];
end

%Choose the calculated meand and variance values
%Word index
existing_words_index = existing_words_index(unique(indices));


%Unique author
unique_author_train = unique(unique_author);
all_author_mean_var = cell(size(unique_author_train,1),1);

for i = 1:size(unique_author_train,1)
    tic
    tmp_txt = train_txt(all_aid(train_ind)==unique_author_train(i),:);
    tmp_txt = tmp_txt';
    tmp_txt = tmp_txt(:);
    tmp_txt = tmp_txt';
    mean_var = zeros(length(existing_words_index),2);
    tmp = zeros(length(existing_words_index),3);
    for j =1:size(existing_words_index,1)
    
       %Takeout the zero element
       tmp_txt=tmp_txt(tmp_txt~=0);
       index = find(tmp_txt==existing_words_index(j));
       %takeout the first, 2nd, last-1 and last element indices
       index(find(index == 1 | index == 2 | index == length(tmp_txt)-1 | index == length(tmp_txt)))=[];
       at1 = word2vec(tmp_txt(index -2),:);
       at2 = word2vec(tmp_txt(index -1),:);
       at3 = word2vec(tmp_txt(index), :);
       at4 = word2vec(tmp_txt(index + 1),:);
       at5 = word2vec(tmp_txt(index + 2),:);
       %Takeout the indices that have word2vec's as ones
       ind1 = find(~ismember(at1, ones(1,300), 'rows'));
       ind2 = find(~ismember(at2, ones(1,300), 'rows'));
       ind4 = find(~ismember(at4, ones(1,300), 'rows'));
       ind5 = find(~ismember(at5, ones(1,300), 'rows'));
       ind_final = intersect(intersect(ind1,ind2),intersect(ind4,ind5));
       at1 = at1(ind_final,:);
       at2 = at2(ind_final,:);
       at3 = at3(ind_final,:);
       at4 = at4(ind_final,:);
       at5 = at5(ind_final,:);
       
       %Cosine distance for the vectors
       dist = (dist_cosine(at1,at3) + dist_cosine(at3,at5))./(dist_cosine(at1,at2) + dist_cosine(at2,at3) +dist_cosine(at3,at4) + dist_cosine(at4,at5));
       dist = dist((~isnan(dist)));
       mean_var(j,1) = mean(dist);
       mean_var(j,2) = var(dist);
       
       if var(dist)==0
           unique_author_train(i)
           existing_words_index(j)
           dist
       end
       
    end
    all_author_mean_var{i} = mean_var;
    unique_author_train(i)
    
    toc
end

fprintf('Training Part is Done and Mean & Variance are Calculated')
label = zeros(size(test_txt,1),2);

for i = 1:size(test_txt,1)
    tmp_txt = test_txt(i,:);
    
    
    all_likely_dist = zeros(size(unique_author_train,1),1);
    for l =1:size(unique_author_train,1)
        mu_sigma = all_author_mean_var{l};
        
        likely_dist = [];
        
        for j =1:size(existing_words_index,1)
           %Takeout the zero element
           tmp_txt=tmp_txt(tmp_txt~=0);
           index = find(tmp_txt==existing_words_index(j));
           %takeout the first, 2nd, last-1 and last element indices
           index(find(index == 1 | index == 2 | index == length(tmp_txt)-1 | index == length(tmp_txt)))=[];
           if isempty(index)
               %Do nothing
           else
               at1 = word2vec(tmp_txt(index -2),:);
               at2 = word2vec(tmp_txt(index -1),:);
               at3 = word2vec(tmp_txt(index), :);
               at4 = word2vec(tmp_txt(index + 1),:);
               at5 = word2vec(tmp_txt(index + 2),:);
               %Takeout the indices that have word2vec's as ones
               ind1 = find(~ismember(at1, ones(1,300), 'rows'));
               ind2 = find(~ismember(at2, ones(1,300), 'rows'));
               ind4 = find(~ismember(at4, ones(1,300), 'rows'));
               ind5 = find(~ismember(at5, ones(1,300), 'rows'));
               ind_final = intersect(intersect(ind1,ind2),intersect(ind4,ind5));
               at1 = at1(ind_final,:);
               at2 = at2(ind_final,:);
               at3 = at3(ind_final,:);
               at4 = at4(ind_final,:);
               at5 = at5(ind_final,:);

               %Cosine distance for the vectors
               dist = (dist_cosine(at1,at3) + dist_cosine(at3,at5))./(dist_cosine(at1,at2) + dist_cosine(at2,at3) +dist_cosine(at3,at4) + dist_cosine(at4,at5));
               dist = dist((~isnan(dist)));
            
               mu = mu_sigma(j,1);
               sigma = mu_sigma(j,2);
               tmp = normpdf(dist,mu,sigma);
               %Take out the ones that there is zero
               likely_dist =[likely_dist; sum(log(tmp(tmp~=0)))];  
           end
               
        end
        all_likely_dist(l) = sum(likely_dist);
        
    end
    %Get the max value index and assign it as a label
    [M,I] = max(all_likely_dist);
    label(i,1) = unique_author_train(I);
    label(i,2) = M;
%     label(i,:)
%     if i>=300
%         break;
%     end
    
    
end



count=0;
Y= all_aid(test_ind);
for i=1:size(test_txt,1)
    if label(i,1)==Y(i)
        count = count +1;
    end
end
Accuracy = count/size(test_txt,1)*100;

result=confusionmatStats(Y, label(:,1));