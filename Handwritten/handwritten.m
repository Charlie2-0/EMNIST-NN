%handwritten
clear all;clc


%load training and cross validation data
load('datasplit.mat');
X = datasplit.train.images;
X = double(X);
CV = datasplit.cv.images;
CV = double(CV);
[m,n] = size(X);
countup=1;


%create labels
Y = datasplit.train.labels;
Y_CV = datasplit.cv.labels;
% define parameters
hidden_layers = 4; %no. of hidden layers
nodes = [30 30 30 30]; %a vector of the number of nodes in each layer
epsilon = 1; 
classes = 26; %output classes
lambda = 0; %reg parameter
funct = @sigmoid; %activation function
functgrad = @sigmoidGrad; %grad of activation function
iter = 400; %number of epochs
lr = 2; %learning rate
for v = 1:100:50000
    Xv = X(1:v,:);
    Yv = Y(1:v,:);
%create network
[theta, y_matrix, t_rows, t_cols,ts] = createThetas(hidden_layers,nodes,n,classes,epsilon,Yv);
init = theta;


%unroll thetas for costFunct
nntheta = [];
for i = 1:hidden_layers+1
nntheta = [nntheta; theta{i}(:)];
end
%find cost & grad
[J, grad] = costFunct(Xv, y_matrix, nntheta, lambda,funct, functgrad,hidden_layers,t_rows,t_cols,ts);
jold = J;
Jcheck = 1000;
i=1
for i = 1:iter
    nntheta = nntheta - lr*grad;
    [J, grad, htheta] = costFunct(Xv, y_matrix, nntheta, lambda,funct, functgrad,hidden_layers,t_rows,t_cols,ts);
    Jcheck = jold-J;
    jold = J;
    Jiter(i)=J;
end
i
% nnCostFunct = @(p) costFunct(X, y_matrix, p, lambda,funct, functgrad,hidden_layers,t_rows,t_cols,ts);
%initial_nn_theta = nntheta;
% options = optimoptions('fminunc','CheckGradients',true,'MaxIter',1,'Display','iter','Algorithm','trust-region','SpecifyObjectiveGradient',true);
% [p, fval] = fmincg(nnCostFunct, nntheta,options)
%test on the training set
%reshape the first theta
theta{1} = reshape(nntheta(1:t_rows(1)* t_cols(1)), [t_rows(1), t_cols(1)]);
%reshape the remaining thetas
for i=2:ts
%[sum(t_rows(1:i-1).*t_cols(1:i-1))+1,sum(t_rows(1:i).*t_cols(1:i)),i]
size(nntheta(sum(t_rows(1:i-1).*t_cols(1:i-1))+1:sum(sum(t_rows(1:i-1).*t_cols(1:i-1)) ...
+(t_rows(i)* t_cols(i)))));
%[t_rows(i), t_cols(i),t_rows(i)*t_cols(i)]
    theta{i} = reshape(nntheta(sum(t_rows(1:i-1).*t_cols(1:i-1))+1:sum(sum(t_rows(1:i-1).*t_cols(1:i-1)) ...
    +(t_rows(i)* t_cols(i)))), [t_rows(i), t_cols(i)]);

end

%feedforward
check = init{1}-theta{1};
[Train_htheta,a,z] = feedforward(Xv,theta,funct,ts);
[J_train, grad] = costFunct(Xv, y_matrix, nntheta, lambda,funct, functgrad,hidden_layers,t_rows,t_cols,ts);
[val_train,loc_train] = max(Train_htheta');
loc_train = loc_train';
counter=0;
for i = 1:length(loc_train)
    if loc_train(i) == Y(i)
       counter=counter+1;
    end
end


[CV_htheta,a,z] = feedforward(CV,theta,funct,ts);
ICV = eye(classes);
CV_matrix = ICV(Y_CV,:);
[J_CV, grad] = costFunct(CV, CV_matrix, nntheta, lambda,funct, functgrad,hidden_layers,t_rows,t_cols,ts);
[val_CV,loc_CV] = max(CV_htheta');
loc_CV = loc_CV';
counterCV=0;
for i = 1:length(loc_CV)
    if loc_CV(i) == Y_CV(i)
       counterCV=counterCV+1;
    end
end
w(countup,1) = counter;
w(countup,2) = J_train;
w(countup,3) = counterCV;
w(countup,4) = J_CV;
w
v
countup = countup+1;
end