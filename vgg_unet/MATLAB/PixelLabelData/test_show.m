img = imread('Label_7.png');
x = unique(img);
size_x = size(x, 1);
img = img*floor(255/size_x);

figure()
imshow(img)

