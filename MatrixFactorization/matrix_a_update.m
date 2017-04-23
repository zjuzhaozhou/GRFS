function A_new = mlt_matrix_a_update(X,lamda_k)
A_new = (X*X')*lamda_k;
coeffient = lamda_k*(X*X')*lamda_k;
A_new = A_new*pinv(coeffient);
end
