function A_new = mlt_matrix_a_update(X,lamda_k)
A_new = (X*X')*lamda_k;%X的行数代表featrue的个数，X的列数代表X的数据个数
coeffient = lamda_k*(X*X')*lamda_k;
A_new = A_new*pinv(coeffient);
end