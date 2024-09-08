clc
clear all
close all

%% Part 1
clc
load('ERP_EEG.mat');

fs = 240;   % Sampling freq
N = 100:100:2500;   % Test numbers
t = 1/fs : 1/fs : length(ERP_EEG(:,1))/fs;  % Time for plotting

% Plotting ERP for each value of N:
for i = 1:length(N)
    figure;
    plot(t, mean(ERP_EEG(:,1:N(i)) ,2));
    grid on
    title("ERP. N = "+N(i));
    xlabel('Time (s)');
    ylabel('Potential (uV)');
    saveas(gcf,"Q1_1_N = "+ N(i) +".jpg");
end
close all

% Plotting main ERP for all values of N:
figure;
for i = 1:length(N)
    plot(t, mean(ERP_EEG(:,1:N(i)) ,2));
    hold on
    grid on
    title("ERP");
    xlabel('Time (s)');
    ylabel('Potential (uV)');
end

%% Part 2
N = 1:2550; %The number of trials for averaging
sample = zeros(size(N)); %Array of maximum absolute values 

%Computing maximum absolute value for each N:
for i = 1:length(N)
    sample(i) = max(abs(mean(ERP_EEG(:,1:N(i)) ,2)));
end

figure;
plot(N, sample);
grid on
title('ERP maximum abs vs N');
xlabel('N');
ylabel('ERP maximum abs');
xlim('tight');

%% Part 3
N = 1:2550;
 
rms_arr = zeros(1, length(N)-1); %Array of RMS values
for i=2:length(N)
    erp_sig1 = mean(ERP_EEG(:, 1:N(i)) ,2); %ERP signal using N trials
    erp_sig2 = mean(ERP_EEG(:, 1:N(i-1)) ,2); %ERP signal using N-1 trials
    rms_arr(i-1) = rms(erp_sig1 - erp_sig2); %Computing RMS
end

figure;
plot(N(2:end), rms_arr);
grid minor
title('RMS vs Number of averaged Trials (i)');
xlabel('i');
ylabel('RMS Value');
xlim('tight');

%% Part 5
N0 = 600;
N = 2550;

% Plotting ERP for different number of trials
figure;
plot(t, mean(ERP_EEG(:, 1:N0),2)); %For N = N0
grid on
legend_str = "N = "+N0; %legend_str: The array of strings which are going to be shown as legend

hold on;
plot(t, mean(ERP_EEG(:, 1:N),2)); %For N = 2550
legend_str = [legend_str, "N = "+N];

plot(t, mean(ERP_EEG(:, 1:round(N0/3)),2)); %For N = N0/3
legend_str = [legend_str, "N = "+round(N0/3)];

N0_arr = randperm(N, N0);
plot(t, mean(ERP_EEG(:, N0_arr),2));  %For N = N0 (randomly chosen)
legend_str = [legend_str, "N = "+N0 + " (random)"];

N0_arr = randperm(N, round(N0/3));
plot(t, mean(ERP_EEG(:, N0_arr),2));  %For N = N0/3 (randomly chosen)
legend_str = [legend_str, "N = "+round(N0/3) + " (random)"];

legend(legend_str); %Setting legend for different values of N

title("ERP for different values of N");
xlabel('Time (s)');
ylabel('Potential (uV)');


