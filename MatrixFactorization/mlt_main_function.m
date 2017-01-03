function lamda_last = mlt_main_function(X,L,alpha,belta,epsilon) 

[N,M] = size(X);
A_0 = rand(N,N); 
lamda_0 = rand_lamda(N);

lamda_new = mlt_lamda_update(X,L,A_0,lamda_0,alpha,belta);
A_new = A_0;
lamda_minus = lamda_new - lamda_0;
lamda_distance = norm(lamda_minus,'fro');


while lamda_distance >= epsilon
    lamda_old = lamda_new;
    lamda_new = mlt_lamda_update(X,L,A_new,lamda_old,alpha,belta);
    A_new = mlt_matrix_a_update(X,lamda_new);
    lamda_minus = lamda_new - lamda_old;
    lamda_distance = norm(lamda_minus,'fro');
end

lamda_last = lamda_new;
disp(lamda_last);
end


function lamda_0 = rand_lamda(N)
lamda_rand_arrow = rand(N,1);
lamda_0 = zeros(N);
for i=1:N
    lamda_0(i,i) = lamda_rand_arrow(i,1);%lamda为对角矩阵，对角线上的元素为介于0、1之间的值
end
disp(lamda_0);
end
