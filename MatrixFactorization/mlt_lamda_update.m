function lamda_new = mlt_lamda_update(X,L,A,lamda_old,alpha,belta)

[N,M] = size(X);
lamda_new = lamda_old; 
Y = X * L * X';
for i = 1:1
    
    gradient_lamda_p_max = mlt_gradientcalu(X,lamda_new,Y,1,A,alpha,belta,M,N);
    gradient_lamda_p_max_index = 1;
    for k = 2:N 
        gradient_lamda_p = mlt_gradientcalu(X,lamda_new,Y,k,A,alpha,belta,M,N); 
        if gradient_lamda_p_max < gradient_lamda_p
            
            gradient_lamda_p_max = gradient_lamda_p;
            gradient_lamda_p_max_index = k;
        end
    end
    p = gradient_lamda_p_max_index;
    if gradient_lamda_p_max > alpha
        thelta_p = -1;
    elseif gradient_lamda_p_max < -(alpha)
        thelta_p = 1;
    else
        thelta_p = 0;
    end
    
    sigma_a_pk_x_pj = belta*Y(p,p);
    for j = 1:M
        for k = 1:N
            sigma_a_pk_x_pj = sigma_a_pk_x_pj + A(p,k)^2 * X(p,j)^2;
        end
    end
    
    
    sigma_r_jk_a_pk_x_pj = -((alpha*thelta_p)/2);
    for j = 1:M
        for k = 1:N
            
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

gradient_lamda_p = 2*belta*Y(p,p)*lamda_new(p,p);
for j = 1:M
   
    for k = 1:N
        r_kj = X(k,j);
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
