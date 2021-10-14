close all;
clear all;
A = importdata('g_theta2_sameopti_keeping.txt');
B = A; %(11:end, 1:2);
i = 1;

vec_x = importdata('vec_x_sameopti_keeping.txt');
vec_xx = vec_x;
%vec_xx = [0.5,0.7; vec_x(2:end, 1:2)];

myVideo = VideoWriter('runs', 'MPEG-4'); %open video file
myVideo.FrameRate = 10;  %can adjust this, 5 - 10 works well for me
open(myVideo)
k = 1;
while i < size(B,1)
    figure(k)
    plot(0.2, 0.1, 'ro', 'MarkerSize',10)
    hold on
    plot(vec_xx(k,1), vec_xx(k,2), "rx", 'MarkerSize',10)
    hold on
    plot(B(i:i+9,1), B(i:i+9,2), 'b+', 'MarkerSize',10)
    grid on
    xlim([-0.1 1.5])
    ylim([-0.1 1.5])
    
    legend('target', 'current TAD sol.', 'samples location')
    hold off
    i = i+10;
    k = k+1;
    pause(0.5) %Pause and grab frame
    frame = getframe(gcf); %get frame
    writeVideo(myVideo, frame);
end
% figure(k+1)
% subplot(1,3,1)
% plot(0.1, 0.15, 'ro', 'MarkerSize',10)
% hold on
% plot(vec_xx(1,1), vec_xx(1,2), "rx", 'MarkerSize',10)
% hold on
% plot(B(1:10,1), B(1:10,2), 'b+', 'MarkerSize',10)
% grid on
% xlim([-0.1 1.5])
% ylim([-0.1 1.5])
%     
% legend('target', 'current TAD sol.', 'samples location')
% hold off
% subplot(1,3,2)
% plot(0.1, 0.15, 'ro', 'MarkerSize',10)
% hold on
% plot(vec_xx(5,1), vec_xx(5,2), "rx", 'MarkerSize',10)
% hold on
% plot(B(51:60,1), B(51:60,2), 'b+', 'MarkerSize',10)
% grid on
% xlim([-0.1 1.5])
% ylim([-0.1 1.5])
%     
% legend('target', 'current TAD sol.', 'samples location')
% hold off
% 
% subplot(1,3,3)
% plot(0.1, 0.15, 'ro', 'MarkerSize',10)
% hold on
% plot(vec_xx(11,1), vec_xx(11,2), "rx", 'MarkerSize',10)
% hold on
% plot(B(101:110,1), B(101:110,2), 'b+', 'MarkerSize',10)
% grid on
% xlim([-0.1 1.5])
% ylim([-0.1 1.5])
%     
% legend('target', 'current TAD sol.', 'samples location')
% hold off

close(myVideo)