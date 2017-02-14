a=zeros([5000, 100], 'single');
z=zeros([5000, 100], 'single');

for i=1:100
    w(:,:,i) = rand(5000, 'single');
end
b(:,100) = rand([5000,100], 'single');
x=rand([5000,1], 'single');

tic
z(:,1) = w(:,:,1)*x+b(:,1);
a(:,1) = 1./(1+exp(-z(:,1)));
for i=2:100
    z(:,i) = w(:, :, i)*a(:,i-1)+b(:,i);
    a(:,i) = 1./(1+exp(-z(:,i)));
end
t=toc