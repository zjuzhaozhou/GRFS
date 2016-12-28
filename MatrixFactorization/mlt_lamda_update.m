function lamda_new = mlt_lamda_update(X,L,A,lamda_old,alpha,belta)
%循环计算函数梯度gradient，（函数为sigma(sigma(x_kj - sigma(lamda_i * a_ik * x_ij))^2) + alpha * |lamda_p| + belta * sigma(Y_pp *  lamda_p * lamda_p)）
%并且根据梯度的计算值，找出梯度最大的lamda值（lamda_p），进行更新，将lamda_p_new = (sigma(sigma(a_pk^2 * x_pj^2)) + belta * Y_pp)^-1 * （sigma(sigma(r_jk * a_pk * x_pj)) - alpha*thelta_p/2）
%n代表所要选取的feature的总数,X为原始数据矩阵，L为拉普拉斯矩阵
%new_lamda = zeros(n,1);
[N,M] = size(X);%X的行数代表featrue的个数，X的列数代表X的数据个数
lamda_new = lamda_old; %lamda_new代表每次更新之后的lamda矩阵
Y = X * L * X';
for i = 1:1
    %迭代100次，每次选取最大的梯度最大的lamda来进行更新
    gradient_lamda_p_max = mlt_gradientcalu(X,lamda_new,Y,1,A,alpha,belta,M,N);%用来暂时存储最大的导数，有可能所有导数均为比0小的负数，那么初始设置的导数就必须为关于lamda_1的导数，且gradient_lamda_p_max_index初始化为1，
                             %防止下面出现gradient_lamda_p_max_index最终经过k的循环后不更新，(即所有的导数均比gradient_lamda_p_max初始值小的情况出现)
    gradient_lamda_p_max_index = 1;%用来暂时存储最大的导数的序号,初始化为1
    for k = 2:N %对于所有的特征进行求导，并且找出最大的导数
        gradient_lamda_p = mlt_gradientcalu(X,lamda_new,Y,k,A,alpha,belta,M,N); %gradient_lamda_p代表每次对于lamda_p求导的结果
        if gradient_lamda_p_max < gradient_lamda_p
            %来找出最大的导数
            gradient_lamda_p_max = gradient_lamda_p;
            gradient_lamda_p_max_index = k;
        end
    end
    p = gradient_lamda_p_max_index;%则用p来代表最大导数的lamda的序号
    if gradient_lamda_p_max > alpha
        thelta_p = -1;
    elseif gradient_lamda_p_max < -(alpha)
        thelta_p = 1;
    else
        thelta_p = 0;
    end
    %对于最大的导数对应的gradient_lamda_p_max_index进行更新
    %先求前面的sigma(sigma(a_pk^2 * x_pj^2)) + belta * Y_pp
    sigma_a_pk_x_pj = belta*Y(p,p);%此处
    for j = 1:M
        for k = 1:N
            sigma_a_pk_x_pj = sigma_a_pk_x_pj + A(p,k)^2 * X(p,j)^2;
        end
    end
    
    %再求后面的sigma(sigma(r_jk * a_pk * x_pj)) - alpha*thelta_p/2
    sigma_r_jk_a_pk_x_pj = -((alpha*thelta_p)/2);
    for j = 1:M
        for k = 1:N
            %先求r_kj
            r_kj = X(k,j);
            if p == 1
                r_kj = r_kj;
            else
                for l = 1:(p-1)
                    r_kj = r_kj - lamda_new(l,l)*A(l,k)*X(l,j);
                end
                for l = (p+1):N
                    r_kj = r_kj - lamda_new(l,l)*A(l,k)*X(l,j);
                end
            end
            sigma_r_jk_a_pk_x_pj = sigma_r_jk_a_pk_x_pj + r_kj*A(p,k)*X(p,j);
        end
    end
    lamda_p_new = (sigma_a_pk_x_pj^-1) * sigma_r_jk_a_pk_x_pj;
    lamda_new(p,p) = lamda_p_new;
    %new_lamda(p,1) = lamda_p_new;
end
end

function gradient_lamda_p = mlt_gradientcalu(X,lamda_new,Y,p,A,alpha,belta,M,N)
%对于lamda_j求导的函数
%X代表原始数据矩阵，lamda_new代表要进行更新的lamda矩阵，Y代表拉普拉斯矩阵与X、X^T矩阵的乘积,j代表要更新的lamda的序号，A代表系数矩阵
gradient_lamda_p = 2*belta*Y(p,p)*lamda_new(p,p);%先加上后面belta * Y_pp *lamda_p^2的关于lamda_p导数
for j = 1:M
   
    for k = 1:N
        r_kj = X(k,j);%r_kj即为论文中的r_kj^-p
        %计算sigma(lamda_i *a_ik*x_ij)(i!=p)
        if p == 1
            r_kj = r_kj;
        else
            for i = 1:(p - 1)
                r_kj = r_kj - lamda_new(i,i)*A(i,k)*X(i,j);
            end
            for i = (p + 1):N
                r_kj = r_kj - lamda_new(i,i)*A(i,k)*X(i,j);
            end
        end
        gradient_lamda_p = gradient_lamda_p + 2*(r_kj - lamda_new(p,p) * A(p,k) * X(p,j)) * (-A(p,k) * X(p,j));
        fprintf('for lamda_p %d: gradient_lamda_p for j:%d,k:%d is %d\n',p,j,k,gradient_lamda_p);
    end
end
end