%resampling is really a problem ? 

load('ml_challenge_author_attribution.mat')
unique_author=unique(train_author);
f1_scores_all=[];
classes_less_f1score_all=[];
classes_less_score_all=[];
for j=1:length(unique_author)
    %Pick the author and subtract it
txt1=train_txt(train_author~=unique_author(j),:);
book1=train_book(train_author~=unique_author(j),:);
aut1=train_author(train_author~=unique_author(j),:);

%Bag of words
bow_train=zeros(size(txt1,1),10000);
for i=1:size(txt1,1)
x=txt1(i,:);
[a,b]=hist(x,unique(x));
bow_train(i,b)=a;
end
%Normalize it
bow_train=bow_train./(sum(bow_train,2)*ones(1,size(bow_train,2)));

bow=bow_train;
for i=1:size(bow,2)
    bow(:,i)=(bow(:,i)-min(bow(:,i)))/max(bow(:,i)-min(bow(:,i)));
end

%Now train svm


indices = crossvalind('Kfold', length(unique(book1)), 5);
f1_scores = zeros(1,5);
book_ids = unique(book1);
classes_less_f1score=zeros(5,10);
classes_less_score=zeros(5,10);
for i=1:5
    
    tic
    train_tr = bow(ismember(book1, (indices~=i).*book_ids), :);

    n_test = bow(ismember(book1, (indices==i).*book_ids), :);

    labels_train = aut1(ismember(book1, (indices~=i).*book_ids), :);
 
    model = train(labels_train, sparse(train_tr),['-s 1', '-C 3'] );
    n_test_label = aut1(ismember(book1, (indices==i).*book_ids), :);

    [predicted_label] = predict(n_test_label, sparse(n_test), model);  
    scores = confusionmatStats(n_test_label, predicted_label);
  
    f1_scores(i) = mean(scores.Fscore)
    [sortedX,sortingIndices] = sort(scores.Fscore,'ascend');
    classes_less_f1score(i,:)=scores.groupOrder(sortingIndices(1:10));
    classes_less_score(i,:)=sortedX(1:10);
    toc
end

f1_scores_all=[f1_scores_all;f1_scores];
classes_less_f1score_all=[classes_less_f1score_all;classes_less_f1score];
classes_less_score_all=[classes_less_score_all;classes_less_score];

j


end


