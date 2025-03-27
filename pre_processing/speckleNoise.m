function [imSpeckle] = speckleNoise(imOriginal, L)
    % imOriginal is the quadratic function of the real image

    [m,n] = size(imOriginal);           % Getting image size
    % [m,n, cp] = size(imOriginal);
    U = gamrnd(L,1/L,[m n]);
    imSpeckle=(imOriginal.*U);
    %figure,imshow(imSpeckle,[])

end