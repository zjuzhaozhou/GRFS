function lamda_last = mlt_main_function_new(X,L,alpha,belta,epsilon) %设置最终退出迭代循环的条件系数为epsilon,L为M*M矩阵
%主函数完成lamda与A的更新，最终将lamda的最终值传出
[N,M] = size(X);%X的行数代表featrue的个数，X的列数代表X的数据个数
A_0 = rand(N,N); %A的行列维数为X的feature数目,A为N*N的每个元素值均为介于0、1之间的一个N阶矩阵
lamda_0 = rand_lamda(N);

i = 1;
lamda_new = mlt_lamda_update_new(X,L,A_0,lamda_0,alpha,belta,i);
A_new = mlt_matrix_a_update(X,lamda_new);%对矩阵A进行一次更新
lamda_minus = lamda_new - lamda_0;
lamda_distance = norm(lamda_minus,'fro');

%循环更新lamda及A，直到lamda_distance小于epsilon
i = 2;
while lamda_distance >= epsilon
    fprintf('the lamda minus caluculate is not good .we go to the next loop of i: %d\n',i);
    lamda_old = lamda_new;
    lamda_new = mlt_lamda_update_new(X,L,A_new,lamda_old,alpha,belta,i);
    fprintf('Now the lamda_new 100 Loop of i : %d has done .we go to the A matrix update\n',i);
    A_new = mlt_matrix_a_update(X,lamda_new);
    fprintf('Now the A matrix update of i : %d has done .we go to the lamda minus caluculate\n',i);
    lamda_minus = lamda_new - lamda_old;
    lamda_distance = norm(lamda_minus,'fro');
    i = i + 1;
end

lamda_last = lamda_new;
disp(lamda_last);
fprintf('Now the final lamda_last has been found. It is like up:\n');
end

function lamda_0 = rand_lamda(N)
lamda_rand_arrow = rand(N,1);
lamda_0 = zeros(N);
for i=1:N
    lamda_0(i,i) = lamda_rand_arrow(i,1);%lamda为对角矩阵，对角线上的元素为介于0、1之间的值
end
disp(lamda_0);
end