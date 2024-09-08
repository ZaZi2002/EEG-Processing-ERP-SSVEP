clc
close all
clear all

%% Part 1-1
load('SSVEP_EEG.mat');

% Bandpass filter between 1 - 40
fs = 250; % Sampling Frequency

[b1,a1] = butter(30,40/(fs/2),'low'); % Butterworth lowpass filter of order 30
[b2,a2] = butter(6,1/(fs/2),'high'); % Butterworth highpass filter of order 6

figure('Name','Part1');
t = 1/fs : 1/fs : length(SSVEP_Signal(1,:))/fs;  % EEG time length

for i = 1:6
    filtered_SSVEP_Signal(i,:) = filter(b1,a1,SSVEP_Signal(i,:));
    filtered_SSVEP_Signal(i,:) = filter(b2,a2,filtered_SSVEP_Signal(i,:));

    subplot(6,2,(2*i-1));
    plot(t,SSVEP_Signal(i,:));
    grid on;
    xlim('tight');
    title(" SSVEP channels " +i);
    xlabel('Time (s)')

    subplot(6,2,(2*i));
    plot(t,filtered_SSVEP_Signal(i,:));
    grid on;
    xlim('tight');
    title(" SSVEP filtered channels " +i);
    xlabel('Time (s)')
end

%% Part 1-2
for i = 1:6
    for j = 1:15
        events_channels(i,j,:) = filtered_SSVEP_Signal(i,Event_samples(j) + 1:Event_samples(j) + 5*fs);
    end
end

%% Part 1-3
t = 0.001 : 1/fs : 5;  % event lengths
for i = 1:15
    figure('Name',"Part3_" +i)
    for j = 1:6
        events_pwelch(j,i,:) = fftshift(pwelch(squeeze(events_channels(j,i,:))));
        N = length(events_pwelch(j,i,:));
        f = fs*(-N/2:N/2-1)/N;
        plot(f,squeeze(db(events_pwelch(j,i,:))));
        hold on;
        xlim([0,40]);
        xlabel('Frequency (Hz)')
        title("Event" + i);
    end
    legend('Channel1','Channel2','Channel3','Channel4','Channel5','Channel6');
    grid minor;
    %saveas(gcf,"Part3_Event" + i + ".png");
end

%% Part 2-1
clc
close all

for i = 1:6
    for j = 1:15
        events_channels(i,j,:) = filtered_SSVEP_Signal(i,Event_samples(j) + 1:Event_samples(j) + 5*fs);
    end
end

%% Part 2-2
clc
t = 1/fs:1/fs:5;

% Frequencies
f1 = 6.5;
f2 = 7.35;
f3 = 8.3;
f4 = 9.6;
f5 = 11.61;

% Events labels
freqs = [1 1 1 2 2 2 3 3 3 4 4 4 5 5 5];

% Defining Yi's based on fi's
y1 = zeros(12,1250);
y2 = zeros(10,1250);
y3 = zeros(8,1250);
y4 = zeros(8,1250);
y5 = zeros(6,1250);
for i = 1:1250
    time = i/fs;
    y1(1,i) = sin(2*pi*f1*time);
    y1(2,i) = cos(2*pi*f1*time);
    y1(3,i) = sin(2*pi*2*f1*time);
    y1(4,i) = cos(2*pi*2*f1*time);
    y1(5,i) = sin(2*pi*3*f1*time);
    y1(6,i) = cos(2*pi*3*f1*time);
    y1(7,i) = sin(2*pi*4*f1*time);
    y1(8,i) = cos(2*pi*4*f1*time);
    y1(9,i) = sin(2*pi*5*f1*time);
    y1(10,i) = cos(2*pi*5*f1*time);
    y1(11,i) = sin(2*pi*6*f1*time);
    y1(12,i) = cos(2*pi*6*f1*time);

    y2(1,i) = sin(2*pi*f2*time);
    y2(2,i) = cos(2*pi*f2*time);
    y2(3,i) = sin(2*pi*2*f2*time);
    y2(4,i) = cos(2*pi*2*f2*time);
    y2(5,i) = sin(2*pi*3*f2*time);
    y2(6,i) = cos(2*pi*3*f2*time);
    y2(7,i) = sin(2*pi*4*f2*time);
    y2(8,i) = cos(2*pi*4*f2*time);
    y2(9,i) = sin(2*pi*5*f2*time);
    y2(10,i) = cos(2*pi*5*f2*time);

    y3(1,i) = sin(2*pi*f3*time);
    y3(2,i) = cos(2*pi*f3*time);
    y3(3,i) = sin(2*pi*2*f3*time);
    y3(4,i) = cos(2*pi*2*f3*time);
    y3(5,i) = sin(2*pi*3*f3*time);
    y3(6,i) = cos(2*pi*3*f3*time);
    y3(7,i) = sin(2*pi*4*f3*time);
    y3(8,i) = cos(2*pi*4*f3*time);

    y4(1,i) = sin(2*pi*f4*time);
    y4(2,i) = cos(2*pi*f4*time);
    y4(3,i) = sin(2*pi*2*f4*time);
    y4(4,i) = cos(2*pi*2*f4*time);
    y4(5,i) = sin(2*pi*3*f4*time);
    y4(6,i) = cos(2*pi*3*f4*time);
    y4(7,i) = sin(2*pi*4*f4*time);
    y4(8,i) = cos(2*pi*4*f4*time);

    y5(1,i) = sin(2*pi*f5*time);
    y5(2,i) = cos(2*pi*f5*time);
    y5(3,i) = sin(2*pi*2*f5*time);
    y5(4,i) = cos(2*pi*2*f5*time);
    y5(5,i) = sin(2*pi*3*f5*time);
    y5(6,i) = cos(2*pi*3*f5*time);
end

% Performing CCA
ro = zeros(15,5);
r = zeros(1,5);
acc = 0;
for i = 1:15
    [~,~,r] = canoncorr(squeeze(events_channels(:,i,:))',y1');
    ro(i,1) = max(r); % ro of CCA with 6.5Hz 
    [~,~,r] = canoncorr(squeeze(events_channels(:,i,:))',y2');
    ro(i,2) = max(r); % ro of CCA with 7.35Hz 
    [~,~,r] = canoncorr(squeeze(events_channels(:,i,:))',y3');
    ro(i,3) = max(r); % ro of CCA with 8.3Hz 
    [~,~,r] = canoncorr(squeeze(events_channels(:,i,:))',y4');
    ro(i,4) = max(r); % ro of CCA with 9.6Hz 
    [~,~,r] = canoncorr(squeeze(events_channels(:,i,:))',y5');
    ro(i,5) = max(r); % ro of CCA with 11.61Hz 
    [max_ro(i),index(i)] = max(ro(i,:));
    if index(i) == freqs(i)
        acc = acc + 1;
    end
end

display("Accuracy of CCA is : " + (acc/15)*100);

%% Part 2-3
% Performing CCA for less channels
for j = 1:15
    ro = zeros(15,5);
    r = zeros(1,5);
    acc = 0;
    for i = 1:j
        [~,~,r] = canoncorr(squeeze(events_channels(:,i,:))',y1');
        ro(i,1) = max(r); % ro of CCA with 6.5Hz 
        [~,~,r] = canoncorr(squeeze(events_channels(:,i,:))',y2');
        ro(i,2) = max(r); % ro of CCA with 7.35Hz 
        [~,~,r] = canoncorr(squeeze(events_channels(:,i,:))',y3');
        ro(i,3) = max(r); % ro of CCA with 8.3Hz 
        [~,~,r] = canoncorr(squeeze(events_channels(:,i,:))',y4');
        ro(i,4) = max(r); % ro of CCA with 9.6Hz 
        [~,~,r] = canoncorr(squeeze(events_channels(:,i,:))',y5');
        ro(i,5) = max(r); % ro of CCA with 11.61Hz 
        [max_ro(i),index(i)] = max(ro(i,:));
        if index(i) == freqs(i)
            acc = acc + 1;
        end
    end
    accuracy(j) = (acc/15)*100;
end

figure;
plot(accuracy);
grid on
xlim tight
xlabel("Number of channels");
ylabel("Accuracy");
title("Accuracy per channels");

%% Part 2-4
% Performing CCA for less samples
for j = 2:1250
    ro = zeros(15,5);
    r = zeros(1,5);
    acc = 0;
    for i = 1:15
        [~,~,r] = canoncorr(squeeze(events_channels(:,i,1:j))',y1(:,1:j)');
        ro(i,1) = max(r); % ro of CCA with 6.5Hz 
        [~,~,r] = canoncorr(squeeze(events_channels(:,i,1:j))',y2(:,1:j)');
        ro(i,2) = max(r); % ro of CCA with 7.35Hz 
        [~,~,r] = canoncorr(squeeze(events_channels(:,i,1:j))',y3(:,1:j)');
        ro(i,3) = max(r); % ro of CCA with 8.3Hz 
        [~,~,r] = canoncorr(squeeze(events_channels(:,i,1:j))',y4(:,1:j)');
        ro(i,4) = max(r); % ro of CCA with 9.6Hz 
        [~,~,r] = canoncorr(squeeze(events_channels(:,i,1:j))',y5(:,1:j)');
        ro(i,5) = max(r); % ro of CCA with 11.61Hz 
        [max_ro(i),index(i)] = max(ro(i,:));
        if index(i) == freqs(i)
            acc = acc + 1;
        end
    end
    accuracy(j) = (acc/15)*100;
end

figure;
plot(accuracy);
grid on
xlim tight
xlabel("Samples");
ylabel("Accuracy");
title("Accuracy per samples");