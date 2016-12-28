clear;
load('COIL20.mat');
nClass = length(unique(gnd));%则nClass为20


%Normalize each data vector to have L2-norm equal to 1 
%对于COIL20.mat中的fea为一个1440*1024的矩阵，其中每一行为一个数据，每一列为一个feature，所以当作我们的函数中的X，fea需要先转置
%对于第一次，先令belta为0，不考虑L矩阵，则L矩阵随意rand一个M*M的，（M为1440）
%返回自己所找的新lamda之后，用新的lamda中找最大的前100个作为X所选取的feature，取这些列作为所要选取的X的列，构成一个新的1440*100的矩阵，形成新的fea,输入到下面的NormalizeFea中。?
L = zeros(1440,1440);
alpha = 0.01;
belta = 0;%先不考虑拉普拉斯矩阵
epsilon = 0.1;%逐渐调整epsilon的值
lamda_last = mlt_main_function_new(fea',L,alpha,belta,epsilon);

%在求出lamda_new之后，要在lamda_new的对角线元素上面找最大的元素，作为所选择的feature的代号
%先把lamda_last中的对角线元素排列成行向量
lamda_last_row = zeros(1,1024);
for i = 1:1024
lamda_last_row(1,i) = lamda_last(i,i);
end

%然后挑出lamda_last_row中最大的前100个元素，先对于lamda_row_last_row升序排序，并保留其原始序号在ind向量中，在找最大（即靠后）的100个序号
[lamda_last_row_sort,ind] = sort(lamda_last_row);
%然后找出ind的后100位的值作为fea所要选取的列，把这些列取出放入一个新矩阵，构成新的fea
fea_new = zeros(1440,100);
for r=1:100
    fea_new(:,ind(end-r+1)) = fea(:,ind(end-r+1));
end
fea_new = NormalizeFea(fea_new);
fprintf('the fea_new is :');
disp(fea_new);
fprintf('the final fea_new which we have selected is like up:\n');

%Clustering in the original space
rand('twister',5489);
label = litekmeans(fea_new,nClass,'Replicates',10);
MIhat = MutualInfo(gnd,label);
disp(['kmeans use all the features. MIhat: ',num2str(MIhat)]);