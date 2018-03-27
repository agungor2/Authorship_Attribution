%%
%%w=10 and s=5 
%Unsupervised Feature Learning
%Divide it by w=10

w=10;
s=5;


load('Vocabulary_wstopwords.mat')
load('ml_challenge_data_wstopwords.mat')
word2vec = csvread('word2vec_data_stop_words.csv');
tfidf_final = load('tfidf_final.mat','tfidf_final');
tfidf_final = tfidf_final.tfidf_final;
unique_author = [39;45;31;21;37];
fprintf('Done uploading files')
%Unique author
unique_author_train = unique(unique_author);

%Split into training and testing part
all_txt =[];
all_aid =[];
all_tfidf_final = [];
for i=1:length(unique_author)
    tmp1 = txt_pieces(aid==unique_author(i),:);
    all_txt = [all_txt; tmp1];
    tmp2 = ones(size(tmp1,1),1)*unique_author(i);
    all_aid = [all_aid; tmp2];
    tmp3 = tfidf_final(aid==unique_author(i),:);
    all_tfidf_final = [all_tfidf_final;tmp3];
end
fprintf('Done extracting authors')

new_txt = [];
new_aid = [];
new_tfidf_final = [];
for i = 1:size(all_txt,1)
   at = all_txt(i,:);
   if ~any(at==0)
   new_txt = [new_txt;at];
   new_aid = [new_aid;all_aid(i)];
   new_tfidf_final = [new_tfidf_final;all_tfidf_final(i,:)];
   end
end
all_txt = new_txt;
all_aid = new_aid;
all_tfidf_final = new_tfidf_final;
%Train and Test Cases seperation
[train_ind, val_ind, test_ind] = dividerand(size(all_txt,1),0.6,0.0,0.4);
train_txt = all_txt(train_ind,:);
test_txt = all_txt(test_ind,:);
train_author = all_aid(train_ind);
test_author = all_aid(test_ind);
train_tfidf_final = all_tfidf_final(train_ind,:);
test_tfidf_final = all_tfidf_final(test_ind,:);
%Now Let's use the w=10 and s=5 to divide the test set
data = zeros(size(all_txt,2)/s*size(all_txt,1)-1,300);

%Now let's define all the text into one straight array
at = all_txt';
at = at(:);
at = at';
%Do the same for tfidfs
at2 = all_tfidf_final';
at2 = at2(:);
at2 = at2';
fprintf('Done creating files')

%find the word2vec indices that are equal to 1
indices = [];
for i =1:10000
    if word2vec(i,:)==zeros(1,300)
        indices = [indices; i];
    end
end

for i =1:size(all_txt,2)/s*size(all_txt,1)-1
    tmp = at(s*(i-1)+1:s*(i+1));
    tmp2 = at2(s*(i-1)+1:s*(i+1));
    %find the elements that are not in the list of indices
    [tmp, ind] = setdiff(tmp,indices);
    %Use the tfidf as weight
    word2vecs = word2vec(tmp,:);
    tfidfs = tmp2(ind);
    tfidfs = repmat(tfidfs',1,300);
    tfidf_word2vec = word2vecs.*tfidfs;
    data(i,:) = mean(tfidf_word2vec); 
    if mod(i,100000)==0
        i
    end
end
fprintf('Done creating data files')

%Take the normr of the data with 1000 batch
batch = 1000;
for i =1:floor(size(data,1)/batch)
    tmp = data((i-1)*batch+1:batch*i,:);
    tmp = normr(tmp);
    data((i-1)*batch+1:batch*i,:) = tmp;
end
%Final batch calculation
tmp = data(floor(size(data,1)/batch)*batch+1:size(data,1),:);
tmp = normr(tmp);
data(floor(size(data,1)/batch)*batch+1:size(data,1),:) = tmp;
fprintf('Done normalization of data file')

%We need to do K-means Clustering for K=1000
numClusters = 1000 ;
[centers, assignments] = vl_kmeans(data', numClusters,'Initialization', 'plusplus','Algorithm', 'ANN');

fprintf('Done with Cluster Calculation')

%%
%Compare the distance for 1000 center points
centers = normc(centers);
train_features = zeros(size(train_txt,1),numClusters);
for i = 1:size(train_txt,1)
    
    at = train_txt(i,:)';
    at = at(:);
    at = at';
    
    at2 = train_tfidf_final(i,:)';
    at2 = at2(:);
    at2 = at2';
    
    data_train = zeros(size(train_txt(i,:),2)/s*size(train_txt(i,:),1)-1, 300);
    for k =1:size(train_txt(i,:),2)/s*size(train_txt(i,:),1)-1
        tmp = at(s*(k-1)+1:s*(k+1));
        tmp2 = at2(s*(k-1)+1:s*(k+1));
        %find the elements that are not in the list of indices
        [tmp, ind] = setdiff(tmp,indices);
        %Use the tfidf as weight
        word2vecs = word2vec(tmp,:);
        tfidfs = tmp2(ind);
        tfidfs = repmat(tfidfs',1,300);
        tfidf_word2vec = word2vecs.*tfidfs;
        data_train(k,:) = mean(tfidf_word2vec);   
        
    end
    data_train = normr(data_train);
    
    for l = 1:size(data_train,1)
        dist = zeros(1,numClusters);
        for j=1:numClusters
            dist(j) = dist_cosine(data_train(l,:),centers(:,j)');
        end
        mean_dist = mean(dist);
        zero_index = find(dist > mean_dist);
        dist(zero_index) = 0;
        train_features(i,:) = train_features(i,:) + dist;
    end
    train_features(i,:) = train_features(i,:) ./size(data_train,1);
    
    i
    
end

%Same steps for the testing data 
test_features = zeros(size(test_txt,1),numClusters);
for i = 1:size(test_txt,1)
    at = test_txt(i,:)';
    at = at(:);
    at = at';
    
    at2 = test_tfidf_final(i,:)';
    at2 = at2(:);
    at2 = at2';
    
    data_test = zeros(size(test_txt(i,:),2)/s*size(test_txt(i,:),1)-1, 300);
    for k =1:size(test_txt(i,:),2)/s*size(test_txt(i,:),1)-1
        tmp = at(s*(k-1)+1:s*(k+1));
        tmp2 = at2(s*(k-1)+1:s*(k+1));
        %find the elements that are not in the list of indices
        [tmp, ind] = setdiff(tmp,indices);
        %Use the tfidf as weight
        word2vecs = word2vec(tmp,:);
        tfidfs = tmp2(ind);
        tfidfs = repmat(tfidfs',1,300);
        tfidf_word2vec = word2vecs.*tfidfs;
        data_test(k,:) = mean(tfidf_word2vec);    
    end
    data_test = normr(data_test);
    
    for l = 1:size(data_test,1)
        dist = zeros(1,numClusters);
        for j=1:numClusters
            dist(j) = dist_cosine(data_test(l,:),centers(:,j)');
        end
        mean_dist = mean(dist);
        zero_index = find(dist > mean_dist);
        dist(zero_index) = 0;
        test_features(i,:) = test_features(i,:) + dist;
    end
    test_features(i,:) = test_features(i,:) ./size(data_test,1);
    i
end

%Murat hoca normalization technique
bow_mfw = train_features;


% Not sure about t6his
bowtst = test_features;
[n, d] = size(train_features);
[n2, d2] = size(test_features);

bowtst=bowtst./(sum(bowtst,2)*ones(1,size(bowtst,2)));
bow_mfw=bow_mfw./(sum(bow_mfw,2)*ones(1,size(bow_mfw,2)));
bow=[bow_mfw;bowtst];
for i=1:size(bow,2)
    bow(:,i)=(bow(:,i)-min(bow(:,i)))/max(bow(:,i)-min(bow(:,i)));
end
bow_train = bow(1:n,:);
bow_test = bow((n+1):end, :);

%Build a Model
model = train(train_author, sparse(bow_train),['-s 1','-C 5']);
[predicted_label] = predict(ones(size(bow_test,1),1), sparse(bow_test), model);

count=0;
Y= test_author;
for i=1:size(test_txt,1)
    if predicted_label(i,1)==Y(i)
        count = count +1;
    end
end
Accuracy = count/size(test_txt,1)*100

result=confusionmatStats(Y, predicted_label(:,1));
mean(result.Fscore)
%Get the elements that are in 50 min
min_elements = cell(100,50);
for i=1:100
    [M,I]=sort(train_features(i,:),2);    
    for j =1:25
        min_elements{i,j} = shortened_vocab{I(j)};
    end
    [M,I]=sort(train_features(i,:),'descend');    
    for j =1:25
        min_elements{i,j+25} = shortened_vocab{I(j)};
    end
end
xlswrite('X.xls',min_elements)