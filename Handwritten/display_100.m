%X = datasplit.train.images;

displaying=[];
disp=[];
vbuffer = 255*ones(28,5);
hbuffer = 255*ones(5,330);
j=1;
h=1;
m=1;

for i = 1:100
    ex = reshape(X(i,:),28,28);
    displaying = [displaying vbuffer ex];
end
for j=1:10
    disp = [disp; hbuffer; displaying(:,m:m+329)];
    m=m+330;
end
imshow(disp)
