%%
%Implement Bag of words
load('Vocabulary_wstopwords.mat')
load('ml_challenge_data_wstopwords.mat')

train_txt = txt_pieces(train_ind,:);
test_txt = txt_pieces(test_ind,:);
train_author = aid(train_ind);
test_author = aid(test_ind);

[n, d] = size(train_txt);
[n2, d2] = size(test_txt);

word_id = 8000:10000;
numvoc = length(word_id);

BOW_train = zeros(n, numvoc);
BOW_test = zeros(n2, numvoc);


%BOW 
for i=1:n
    BOW_train(i,:) = histc(train_txt(i, :), word_id);
end

for i=1:n2
    BOW_test(i,:) = histc(test_txt(i, :), word_id);
end


%BOW authorwise
author_list = containers.Map;
unique_author_list = unique(train_author);
for i = 1:length(unique(train_author))
    tmp_txt = train_txt(train_author==unique_author_list(i), :);
    a = unique(tmp_txt);
    out = [a,histc(tmp_txt(:),a)];
    %Take the most occurent 20 words for each author
    %write it to a cell
    [~,idx] = sort(out(:,2),'descend'); % sort just the first column
    sortedmat = out(idx,:);   % sort the whole matrix using the sort indices
    author_list(int2str(i)) = sortedmat(1:20,:);
    authorwise_top_elements = "";
    for j=1:20
        authorwise_top_elements = authorwise_top_elements + " " +shortened_vocab{sortedmat(j,1)};
    end
    authorwise_top_elements
end

%%