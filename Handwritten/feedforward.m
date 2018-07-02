function [htheta,a,z] = feedforward(X,theta,funct,ts)

%X is the input vector where each row is a feature vector
%theta is a cell array where each entry is a matrix that is theta1, theta2, etc
%nodes is a vector where each entry corrosponds to the number of nodes in
%each subsequent hidden layer
%class is the number of classes and hence the number of nodes in the ouput
%layer



%run the first layer
[r,c] = size(X);
a{1} = [ones(r, 1) X]; %(87360x785);
%z{2} = (theta{1}*a{1}')'; %*(4x785)*(785x87360) = 4x87360;
z{1} = [];
%run the rest of the layers and store to cell array
for i = 2:ts
    %[size(theta{i-1}), size(a{i-1}')]
    z{i} = (theta{i-1}*a{i-1}')';
    [r,c] = size(z{i});
    a{i} = [ones(r, 1) funct(z{i})];

end
%z{ts+1} = (theta{ts}*a{ts}')';
%store result of the last layer to htheta
%[h,zs] = size(z);
[r,c] = size(z{ts});
htheta = funct(theta{ts}*a{ts}')';

end
