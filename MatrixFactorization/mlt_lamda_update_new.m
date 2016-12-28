function lamda_new = mlt_lamda_update_new(X,L,A,lamda_old,alpha,belta,o)
%循环计算函数梯度gradient，（函数为sigma(sigma(x_kj - sigma(lamda_i * a_ik * x_ij))^2) + alpha * |lamda_p| + belta * sigma(Y_pp *  lamda_p * lamda_p)）
%并且根据梯度的计算值，找出梯度最大的lamda值（lamda_p），进行更新，将lamda_p_new = (sigma(sigma(a_pk^2 * x_pj^2)) + belta * Y_pp)^-1 * （sigma(sigma(r_jk * a_pk * x_pj)) - alpha*thelta_p/2）
%n代表所要选取的feature的总数,X为原始数据矩阵，L为拉普拉斯矩阵
%new_lamda = zeros(n,1);
[N,M] = size(X);%X的行数代表featrue的个数，X的列数代表X的数据个数
lamda_new = lamda_old; %lamda_new代表每次更新之后的lamda矩阵
Y = X * L * X';

%对于A_i_X_i和lamda_i_A_i_X_i矩阵向量的求解，因为A和X,lamda都是固定的，所以放在mlt_lamda_update函数中，并把结果传入mlt_gradientcalu函数中，而不用每次都重复计算了,
%其中A_i_X_i放在for z循环的外面，而不用每次都要在更新梯度后再次计算一遍，且lamda_i_A_i_X_i的每一个元素也只需要用lamda_new(i,i) * A_i_X_i{1,i}，而不用再算一遍X(i,:)'*A(i,:)
A_i_X_i = cell(1,N);%创建空的元细胞数组，来存储N个矩阵，（X的第i行的转置与A的第i行相乘所得的M*N矩阵）
for i=1:N
    A_i_X_i{1,i}= X(i,:)' * A(i,:);
end

%对于后面更新具有最大梯度的lamda中，sigma(a_pk * x_pj)^2也可以提前存入矩阵中，存1*1024维矩阵中，矩阵元素分别对应于每一个lamda_p的sigma(sigma(a_pk^2 * x_pj^2)) + belta * Y_pp
sigma_a_pk_x_pj_square = zeros(1,N);
for p=1:N
    sigma_a_pk_x_pj = belta*Y(p,p);%此处求belta * Y_pp
    for j = 1:M
        for k = 1:N
            sigma_a_pk_x_pj = sigma_a_pk_x_pj + A_i_X_i{1,p}(j,k)^2;
        end
    end
    sigma_a_pk_x_pj_square(1,p) = sigma_a_pk_x_pj;
end

lamda_i_A_i_X_i = cell(1,N);%创建空的元细胞数组，来存储N个矩阵，（X的第i行的转置与A的第i行相乘所得的矩阵，再与lamda_i这个数（即lamda_new(i,i)）相乘所得M*N矩阵）
fprintf('Now we are calulating lamda_i_A_i_X_i\n');
for i=1:N
    lamda_i_A_i_X_i{1,i} = lamda_old(i,i) * A_i_X_i{1,i};
end

%对于sigma(lamda_i * a_i_k * x_i_j)(i=1~N) (共M*N个)要存储起来，存储为一个矩阵，
%则计算对于每个lamda_p的梯度时，可以直接取出值，（即矩阵的第k,j号元素），而不用每次求关于不同lamda_p的梯度时，都要重新求一遍值
%此处求是为了z=1时的lamda梯度计算进行求解的
fprintf('Now we are calulating x_kj_minus_sigma_lamda_A_ik_X_ij\n');
x_kj_minus_sigma_lamda_A_ik_X_ij = zeros(N,M); %用来存储x_ij - sigma(lamda_i * a_i_k * x_i_j)
for j = 1:M
    for k = 1:N
        r_kj = X(k,j);%r_kj即为论文中的x_kj - sigma(lamda_i * a_i_k * x_i_j),没有去除lamda_p * a_pk * x_pj
        for i=1:N
            r_kj = r_kj - lamda_i_A_i_X_i{1,i}(j,k);
        end
        x_kj_minus_sigma_lamda_A_ik_X_ij(k,j) = r_kj;
    end
end

p = 0;%初始化p，使p不会在每一次while循环后都被抹去

for z = 1:100
    %迭代100次，每次选取最大的梯度最大的lamda来进行更新
    lamda_old = lamda_new;%来让上次循环的lamda_new赋值为这次的lamda_old
    % lamda_i_A_i_X_i = cell(1,N);%创建空的元细胞数组，来存储N个矩阵，（X的第i行的转置与A的第i行相乘所得的矩阵，再与lamda_i这个数（即lamda_new(i,i)）相乘所得M*N矩阵）
    fprintf('Now we are calulating lamda_i_A_i_X_i\n');
    %每次循环仅仅更新了一个lamda_p,其余lamda都没有变，所以可以仅仅将上一次计算出的lamda_i_A_i_X_i这个元细胞数组的lamda_i_A_i_X_i{1,p}元素进行改变，而不用再次将所有都重新计算一遍
    % for i=1:N
        % lamda_i_A_i_X_i{1,i} = lamda_new(i,i) * A_i_X_i{1,i};
    % end
    if z ~= 1
        lamda_i_A_i_X_i{1,p} = lamda_old(p,p) * A_i_X_i{1,p}; %仅仅将上一次计算出的lamda_i_A_i_X_i这个元细胞数组的lamda_i_A_i_X_i{1,p}元素进行改变，而不用再次将所有都重新计算一遍
    end
    
    %对于sigma(lamda_i * a_i_k * x_i_j)(i=1~N) (共M*N个)要存储起来，存储为一个矩阵，
    %则计算对于每个lamda_p的梯度时，可以直接取出值，（即矩阵的第k,j号元素），而不用每次求关于不同lamda_p的梯度时，都要重新求一遍值
    fprintf('Now we are calulating x_kj_minus_sigma_lamda_A_ik_X_ij\n');
    %与上面求lamda_i_A_i_X_i时相同，此时仅仅一个lamda在上一次的z的循环中有所改变，所以仅仅改变x_kj_minus_sigma_lamda_A_ik_X_ij对应于lamda_p的值即可，而x_kj_minus_sigma_lamda_A_ik_X_ij矩阵其他的值不用变
    % x_kj_minus_sigma_lamda_A_ik_X_ij = zeros(N,M); %用来存储x_ij - sigma(lamda_i * a_i_k * x_i_j)
    % for j = 1:M
        % for k = 1:N
            % r_kj = X(k,j);%r_kj即为论文中的x_kj - sigma(lamda_i * a_i_k * x_i_j),没有去除lamda_p * a_pk * x_pj，包括lamda_p * a_pk * x_ij也被减掉了
            % for i=1:N
                % r_kj = r_kj - lamda_i_A_i_X_i{1,i}(j,k);
            % end
            % x_kj_minus_sigma_lamda_A_ik_X_ij(k,j) = r_kj;
        % end
    % end
    %因为每次循环过后，lamda矩阵都会发生变化，所以x_kj_minus_sigma_lamda_A_ik_X_ij矩阵中的每个值都会由于所少减去的lamda_p*a_ik_x_ij发生了变化而改变，所以x_kj_minus_sigma_lamda_A_ik_X_ij的每个值都要更新，但是中间可以少一个for循环（即求r_kj不用再循环减了，而是可以先加上次的lamda_p*a_ik_x_ij，再减这次的lamda_p*a_ik_x_ij）
    if z~=1
        for j = 1:M
            for k = 1:N
                r_kj = x_kj_minus_sigma_lamda_A_ik_X_ij(k,j);%r_kj即为论文中的x_kj - sigma(lamda_i * a_i_k * x_i_j),没有去除lamda_p * a_pk * x_pj
                r_kj = r_kj + lamda_p_old_time * A_i_X_i{1,p}(j,k) - lamda_p_new * A_i_X_i{1,p}(j,k);
                x_kj_minus_sigma_lamda_A_ik_X_ij(k,j) = r_kj;
            end
        end
    end
    gradient_lamda_p_max = mlt_gradientcalu_new(X,lamda_old,Y,1,A,alpha,belta,M,N,A_i_X_i,x_kj_minus_sigma_lamda_A_ik_X_ij);%用来暂时存储最大的导数，有可能所有导数均为比0小的负数，那么初始设置的导数就必须为关于lamda_1的导数，且gradient_lamda_p_max_index初始化为1，
                             %防止下面出现gradient_lamda_p_max_index最终经过k的循环后不更新，(即所有的导数均比gradient_lamda_p_max初始值小的情况出现)?
    gradient_lamda_p_max_index = 1;%用来暂时存储最大的导数的序号,初始化为1
    fprintf('Now the time of mlt_lamda_update_new is : %d ;the loop of z in 100 is : %d ; the gradient_lamda_p of p: 1 is %d\n',o,z,gradient_lamda_p_max);
    for k = 2:N %对于所有的特征进行求导，并且找出最大的导数?
        gradient_lamda_p = mlt_gradientcalu_new(X,lamda_old,Y,k,A,alpha,belta,M,N,A_i_X_i,x_kj_minus_sigma_lamda_A_ik_X_ij); %gradient_lamda_p代表每次对于lamda_p求导的结果
        fprintf('Now the time of mlt_lamda_update_new is : %d ; the loop of z in 100 is : %d ; the gradient_lamda_p of p: %d is %d\n',o,z,k,gradient_lamda_p);
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
    sigma_a_pk_x_pj = sigma_a_pk_x_pj_square(1,p);
    
    
    %再求后面的sigma(sigma(r_jk * a_pk * x_pj)) - alpha*thelta_p/2，因为每次循环过后，lamda矩阵都会发生变化，所以sigma(sigma(r_jk * a_pk * x_pj))都需要重新计算
    sigma_r_jk_a_pk_x_pj = -((alpha*thelta_p)/2);
    for j = 1:M
        for k = 1:N
            r_kj = x_kj_minus_sigma_lamda_A_ik_X_ij(k,j) + lamda_old(p,p)*A_i_X_i{1,p}(j,k) ; %x_kj_minus_sigma_lamda_A_ik_X_ij(k,j)将lamda_p*a_pk*x_pj也给减掉了，所以要在求r_kj时再加回来
            sigma_r_jk_a_pk_x_pj = sigma_r_jk_a_pk_x_pj + r_kj*A_i_X_i{1,p}(j,k); 
        end
    end
    lamda_p_new = (sigma_a_pk_x_pj^-1) * sigma_r_jk_a_pk_x_pj;
    lamda_p_old_time = lamda_old(p,p);%存储本次更新的lamda_p在更新前的值，用来在下一次的x_kj_minus_sigma_lamda_A_ik_X_ij的每个元素的求解中加上lamda_p_old_time，再减去lamda_p_new
    lamda_new = lamda_old;
    lamda_new(p,p) = lamda_p_new;
    %new_lamda(p,1) = lamda_p_new;
    for j=1:N
        if lamda_old(j,j) ~= lamda_new(j,j) && j ~= p %对于lamda中的每个值进行比对如果不仅仅改变了一个lamda(p,p)的值，则报错，看看到底是哪次让所有lamda的对角元素值全部赋值为0
        % &&符号为“与”，如果前面为“0”，则直接将整个逻辑表达式返回值为“0”，而不会再求&&后面的半个逻辑表达式的值了
            fprintf('Now the lamda has changed value :%d , and the value has changed not for p: %d . Now the z is :%d\n',j,p,z);
        end
    end
    disp(lamda_new);
    fprintf('Now the time of mlt_lamda_update_new is : %d ; and the loop of all lamda gradient is over. the lamda_new of z: %d is in the up. \n',o,z);
    if lamda_new(p,p) == 0
        fprintf('Now the time of mlt_lamda_update_new is : %d ; and the loop of all lamda gradient is over. the lamda_new of z: %d is in the up. And the lamda_new(%d,%d) is zero\n',o,z,p,p);%检测是p为何值时，让lamda_new中出现0的
    end

end
end

function gradient_lamda_p = mlt_gradientcalu_new(X,lamda_new,Y,p,A,alpha,belta,M,N,A_i_X_i,x_kj_minus_sigma_lamda_A_ik_X_ij)
%对于lamda_j求导的函数
%X代表原始数据矩阵，lamda_new代表要进行更新的lamda矩阵，Y代表拉普拉斯矩阵与X、X^T矩阵的乘积,j代表要更新的lamda的序号，A代表系数矩阵
gradient_lamda_p = 2*belta*Y(p,p)*lamda_new(p,p);%先加上后面belta * Y_pp *lamda_p^2的关于lamda_p导数
[N,M] = size(X);

%分别用X矩阵的第i行行向量的转置和A矩阵的第i行行向量相乘，得一个M*N的矩阵，
%矩阵中的元素为(x_i_1*a_i_1, x_i_1*a_i_2, x_i_1*a_i_3, …… , x_i_1*a_i_n; x_i_2*a_i_1, x_i_2*a_i_2, ……, x_i_2*a_i_n; …………; x_i_m*a_i_1,x_i_m*a_i_2,…… ,x_i_m*a_i_n)
%则对于第i行的X矩阵与A矩阵的元素乘积可以从这N个M*N的矩阵中所取得，只要存储下这N个矩阵，便可以从中取出元素来代替每次需要乘出A(i,k)*X(i,j)
%同理，再在A矩阵的第i行行向量和X矩阵的第i行行向量的转置乘出的矩阵前面乘上lamda_i，得N个矩阵，便可以lamda_new(i,i)*A(i,k)*X(i,j)也不用每次算出，而从矩阵中取出元素即可。
%lamda_x_a矩阵为（lamda_i*x_i_1*a_i_1,lamda_i*x_i_1*a_i_2,……,lamda_i*x_i_1*a_i_n;lamda_i*x_i_2*a_i_1,lamda_i*x_i_2*a_i_2,……,lamda_i*x_i_2*a_i_n; ………… ; lamda_i*x_i_m*a_i_1,lamda_i*x_i_m*a_i_2,……lamda_i*x_i_m*a_i_n）

%对于A_i_X_i矩阵向量的求解，因为A和X都是固定的，所以放在mlt_lamda_update函数中，并把结果传入mlt_gradientcalu函数中，而不用每次都重复计算了
% A_i_X_i = cell(1,N);%创建空的元细胞数组，来存储N个矩阵，（X的第i行的转置与A的第i行相乘所得的矩阵）
% for i=1:N
    % A_i_X_i{1,i}= X(i,:)' * A(i,:);
% end

% lamda_i_A_i_X_i = cell(1,N);%创建空的元细胞数组，来存储N个矩阵，（X的第i行的转置与A的第i行相乘所得的矩阵，再与lamda_i（即lamda_new(i,i)）相乘所得矩阵）
% for i=1:N
    % lamda_i_A_i_X_i{1,i} = lamda_new(i,i) * X(i,:)'*A(i,:);
% end

for j = 1:M
    for k = 1:N
        gradient_lamda_p = gradient_lamda_p + 2 * x_kj_minus_sigma_lamda_A_ik_X_ij(k,j) * (-A_i_X_i{1,p}(j,k));
    end
end
end