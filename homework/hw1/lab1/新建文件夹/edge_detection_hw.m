clc;
clear;
close all;

% Read input image
I = imread('test2.jpg');

% Convert to grayscale if needed
if size(I, 3) == 3
    gray = rgb2gray(I);
else
    gray = I;
end

% Three edge detectors
E_sobel   = edge(gray, 'sobel');
E_prewitt = edge(gray, 'prewitt');
E_canny   = edge(gray, 'canny');

% Display results
figure;

subplot(2,2,1);
imshow(I);
title('Original Image');

subplot(2,2,2);
imshow(E_sobel);
title('Sobel');

subplot(2,2,3);
imshow(E_prewitt);
title('Prewitt');

subplot(2,2,4);
imshow(E_canny);
title('Canny');