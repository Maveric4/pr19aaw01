img = imread('K_000_005.PNG');
x = unique(img);
size_x = size(x, 1);
img = img*floor(255/size_x);

figure()
imshow(img)

