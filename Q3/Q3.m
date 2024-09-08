clc
close all
clear all

%% Part 1
load('Ex3.mat');
fs = 256;   % Sampling frequency
N = size(TrainData,3);  % Number of tests

% Zero mean data
TrainData = TrainData - mean(TrainData,2);
TestData = TestData - mean(TestData,2);

% Seperating each class
t1 = 0;
t2 = 0;
for i = 1:N
    if (TrainLabel(i) == 1)
        t1 = t1 + 1;
        class1(:,:,t1) = TrainData(:,:,i);
    else
        t2 = t2 + 1;
        class2(:,:,t2) = TrainData(:,:,i);
    end
end

% Sum of covariance matrixes of each channel
C1 = zeros(30,30);
C2 = zeros(30,30);
for i = 1:t1
    C1 = C1 + class1(:,:,i)*class1(:,:,i).';
end
for i = 1:t2
    C2 = C2 + class2(:,:,i)*class2(:,:,i).';
end
C1 = C1/t1;
C2 = C2/t2;

% GEVD
[W,lambda] = eig(C1,C2);
[~,perm] = sort(diag(lambda),'descend');
W = W(:,perm);
lambda = lambda(perm,perm);

% Filtering
W_CSP = [W(:,1) W(:,30)];
filtered_data = transpose(W_CSP)*reshape(TrainData,30,256*165);

% Plotting
n = 4; % Number of tests
t = 1/fs:1/fs:n;
figure;
subplot(2,1,1);
plot(t,filtered_data(1,1:n*fs));
xlim tight;
title("Channel1");
xlabel("Time(s)");
grid minor

subplot(2,1,2);
plot(t,filtered_data(2,1:n*fs));
xlim tight;
title("Channel2");
xlabel("Time(s)");
grid minor

%% Part 2
% clc
% load("AllElectrodes.mat");
% Z = zeros(64,2);
% Z(1:30,:) = Z(1:30,:) + W_CSP;
% for i = 1:64
%     X(i) = AllElectrodes.X;
%     Y(i) = AllElectrodes.Y;
%     L(i,:) = AllElectrodes.labels;
% end
% figure('Name','Topomap for channel1')
% plottopomap(X.',Y.',L ,Z(:,1));
% title("Topomap for Data1 Source " + n);


%% Part 3
% Folds making
fold_numbers = 41;
fold1 = TrainData(:,:,1:fold_numbers);
fold2 = TrainData(:,:,fold_numbers+1:2*fold_numbers);
fold3 = TrainData(:,:,2*fold_numbers+1:3*fold_numbers);
fold4 = TrainData(:,:,3*fold_numbers+1:4*fold_numbers);
fold1_labels = TrainLabel(1:fold_numbers);
fold2_labels = TrainLabel(fold_numbers+1:2*fold_numbers);
fold3_labels = TrainLabel(2*fold_numbers+1:3*fold_numbers);
fold4_labels = TrainLabel(3*fold_numbers+1:4*fold_numbers);

% Seperating train & test
Data1_train = zeros(30,256,fold_numbers*3);
Data1_train(:,:,1:fold_numbers) = fold1;
Data1_train(:,:,fold_numbers+1:2*fold_numbers) = fold2;
Data1_train(:,:,2*fold_numbers+1:3*fold_numbers) = fold3;
Data1_test = fold4;
Label1_train = zeros(1,fold_numbers*3);
Label1_train(1:fold_numbers) = fold1_labels;
Label1_train(fold_numbers+1:2*fold_numbers) = fold2_labels;
Label1_train(2*fold_numbers+1:3*fold_numbers) = fold3_labels;
Label1_test = fold4_labels;

Data2_train = zeros(30,256,fold_numbers*3);
Data2_train(:,:,1:fold_numbers) = fold2;
Data2_train(:,:,fold_numbers+1:2*fold_numbers) = fold3;
Data2_train(:,:,2*fold_numbers+1:3*fold_numbers) = fold4;
Data2_test = fold1;
Label2_train = zeros(1,fold_numbers*3);
Label2_train(1:fold_numbers) = fold2_labels;
Label2_train(fold_numbers+1:2*fold_numbers) = fold3_labels;
Label2_train(2*fold_numbers+1:3*fold_numbers) = fold4_labels;
Label2_test = fold1_labels;

Data3_train = zeros(30,256,fold_numbers*3);
Data3_train(:,:,1:fold_numbers) = fold1;
Data3_train(:,:,fold_numbers+1:2*fold_numbers) = fold3;
Data3_train(:,:,2*fold_numbers+1:3*fold_numbers) = fold4;
Data3_test = fold2;
Label3_train = zeros(1,fold_numbers*3);
Label3_train(1:fold_numbers) = fold1_labels;
Label3_train(fold_numbers+1:2*fold_numbers) = fold3_labels;
Label3_train(2*fold_numbers+1:3*fold_numbers) = fold4_labels;
Label3_test = fold2_labels;

Data4_train = zeros(30,256,fold_numbers*3);
Data4_train(:,:,1:fold_numbers) = fold1;
Data4_train(:,:,fold_numbers+1:2*fold_numbers) = fold2;
Data4_train(:,:,2*fold_numbers+1:3*fold_numbers) = fold4;
Data4_test = fold3;
Label4_train = zeros(1,fold_numbers*3);
Label4_train(1:fold_numbers) = fold1_labels;
Label4_train(fold_numbers+1:2*fold_numbers) = fold2_labels;
Label4_train(2*fold_numbers+1:3*fold_numbers) = fold4_labels;
Label4_test = fold3_labels;

W_number = 15; % Number of filters
for i = 1:W_number
    % Filtered data
    [Filtered1_train ,Filtered1_test] = filtering(Data1_train,Label1_train,Data1_test,fold_numbers*3,fold_numbers,i);
    [Filtered2_train ,Filtered2_test] = filtering(Data2_train,Label2_train,Data2_test,fold_numbers*3,fold_numbers,i);
    [Filtered3_train ,Filtered3_test] = filtering(Data3_train,Label3_train,Data3_test,fold_numbers*3,fold_numbers,i);
    [Filtered4_train ,Filtered4_test] = filtering(Data4_train,Label4_train,Data4_test,fold_numbers*3,fold_numbers,i);
    
    % Feature extraction
    features1_train = featureExtraction(Filtered1_train,i);
    features2_train = featureExtraction(Filtered2_train,i);
    features3_train = featureExtraction(Filtered3_train,i);
    features4_train = featureExtraction(Filtered4_train,i);
    features1_test = featureExtraction(Filtered1_test,i);
    features2_test = featureExtraction(Filtered2_test,i);
    features3_test = featureExtraction(Filtered3_test,i);
    features4_test = featureExtraction(Filtered4_test,i);

    % KNN
    Neighbor_number = 7;
    KNN_model1 = fitcknn(features1_train,Label1_train,'NumNeighbors',Neighbor_number,'Distance','euclidean');
    KNN_model2 = fitcknn(features2_train,Label2_train,'NumNeighbors',Neighbor_number,'Distance','euclidean');
    KNN_model3 = fitcknn(features3_train,Label3_train,'NumNeighbors',Neighbor_number,'Distance','euclidean');
    KNN_model4 = fitcknn(features4_train,Label4_train,'NumNeighbors',Neighbor_number,'Distance','euclidean');
    predicted_labels_knn1 = predict(KNN_model1,features1_test);
    predicted_labels_knn2 = predict(KNN_model2,features2_test);
    predicted_labels_knn3 = predict(KNN_model3,features3_test);
    predicted_labels_knn4 = predict(KNN_model4,features4_test);

    accuracy = zeros(4,1);
    % Computing accuracy of models
    accuracy(1,:) = sum(predicted_labels_knn1.' == Label1_test) / numel(Label1_test);
    accuracy(2,:) = sum(predicted_labels_knn2.' == Label2_test) / numel(Label2_test);
    accuracy(3,:) = sum(predicted_labels_knn3.' == Label3_test) / numel(Label3_test);
    accuracy(4,:) = sum(predicted_labels_knn4.' == Label4_test) / numel(Label4_test);
    mean_accuracy(i) = mean(accuracy,1); 
end
% Max accuracy
[max_acc,best_filter_number] = max(mean_accuracy);
disp("Maximum accuracy is = " + max_acc*100 + "%     with " + best_filter_number + " filters.")


%% Part 4
filter_number = best_filter_number;
% Filtering
[Filtered_train ,Filtered_test] = filtering(TrainData,TrainLabel,TestData,size(TrainData,3),size(TestData,3),filter_number);

% Feature extraction
features_train = featureExtraction(Filtered_train,filter_number);
features_test = featureExtraction(Filtered_test,filter_number);

% KNN model training
KNN_model = fitcknn(features_train,TrainLabel,'NumNeighbors',Neighbor_number,'Distance','euclidean');
predicted_labels_knn = predict(KNN_model,features_test);

% Saving test labels
save('TestLabel.mat',"predicted_labels_knn");

%% Functions
function [filtered_train,filtered_test] = filtering(TrainData,TrainLabel,TestData,train_number,test_number,W_number)
    t1 = 0;
    t2 = 0;
    for i = 1:train_number
        if (TrainLabel(i) == 1)
            t1 = t1 + 1;
            class1(:,:,t1) = TrainData(:,:,i);
        else
            t2 = t2 + 1;
            class2(:,:,t2) = TrainData(:,:,i);
        end
    end
    
    % Sum of covariance matrixes of each channel
    C1 = zeros(30,30);
    C2 = zeros(30,30);
    for i = 1:t1
        C1 = C1 + class1(:,:,i)*class1(:,:,i).';
    end
    for i = 1:t2
        C2 = C2 + class2(:,:,i)*class2(:,:,i).';
    end
    C1 = C1/t1;
    C2 = C2/t2;
    
    % GEVD
    [W,lambda] = eig(C1,C2);
    [~,perm] = sort(diag(lambda),'descend');
    W = W(:,perm);
    lambda = lambda(perm,perm);
    
    % Filtering
    W_CSP = zeros(30,W_number*2);
    for i = 1:W_number
        W_CSP(:,i) = W(:,i);
        W_CSP(:,2*W_number+1-i) = W(:,31-i);
    end
    for i = 1:train_number
        filtered_train(:,:,i) = transpose(W_CSP)*TrainData(:,:,i);
    end
    for i = 1:test_number
        filtered_test(:,:,i) = transpose(W_CSP)*TestData(:,:,i);
    end
end
function [features] = featureExtraction(Data,W_number)
    N = size(Data,3);
    features = zeros(N,W_number*2);
    for i = 1:N
        for j = 1:W_number*2
            features(i,j) = log(var(Data(j,:,i)));
        end
    end
end

