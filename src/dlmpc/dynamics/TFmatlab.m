
%function [t,y] = SystemPlant(u,t,delay)
%num=[3];
%den=[250, 35, 1];
%system = tf(num,den,'ioDelay',delay);
%[y, t] = lsim(system, u, t);
%end


function [t,y] = SystemPlant(u,t,Ts=1,c2d=True)
num=[3];
den=[250, 35, 1];
g=tf(num,den,'ioDelay',20)
if c2d
   system = c2d(g, Ts)
end


t = 0:Ts:100; % 时间范围从0到10秒，步长为采样时间
u = ones(size(t)); % 假设输入是一个恒定的信号

% 对系统进行仿真
% 如果是连续时间系统，使用lsim
[y, t] = lsim(system_discrete, u, t);
[y1, t1] = lsim(system, u, t);

% 如果是离散时间系统，使用dsimul
% y = dsimul(system_discrete, u);

% 绘制输出结果
plot(t, y);
xlabel('Time [s]');
ylabel('Output [y]');
% title('Output of the Transfer Function 3/(s+1)');
hold on;
plot(t1, y1);