function [theta, y_matrix, t_rows, t_cols,ts] = createThetas(hid_lay, nodes, features, class, epsilon, Y);
% hid_lay is the number of hidden layers

% nodes is a vector of integers containing the number of nodes in each
% hidden layer

% features is the number of input features

% class is the number of classes

%create the first theta
theta{1} = rand(nodes(1), 1 + features) * 2 * epsilon - epsilon;

%create the rest of the thetas
nodes = [nodes];
for i = 2:hid_lay
    theta{i} = rand(nodes(i), 1 + nodes(i-1)) * 2 * epsilon - epsilon;
end

%do the last one

theta{i+1} = rand(class, 1 + nodes(i))* 2 * epsilon - epsilon;

%create the answer matrix

I = eye(class);
y_matrix = I(Y,:);


%[x_row,x_col] = size(X); %x_row is the number of features and hence the size of the input layer
[b, ts] = size(theta);
t_rows = [];
t_cols = [];

%create a vector of ref values for each layer
 for i = 1:ts
 
     [rows, cols] = size(theta{i});
     t_rows = [t_rows;rows];
     t_cols = [t_cols;cols];
 
 end
end