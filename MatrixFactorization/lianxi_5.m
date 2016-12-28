function lamda_0 = lianxi_5(N)
lamda_rand_arrow = rand(N,1);
lamda_0 = zeros(N);
for i=1:N
    lamda_0(i,i) = lamda_rand_arrow(i,1);
end
disp(lamda_0);
end