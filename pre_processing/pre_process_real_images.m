%% No Complex Images

% if the image is in intensity --> take the sqrt (amplitude image)

base_path = '.../dataset_no_SLC/mat/';
filename = 'cascine_1L.mat';

im = imread(fullfile(base_path,filename));

% compute the 99 percentile
p = prctile(im(:), 99); 

% divide im for p (I can also clip and then divide for p)
im_01_to_be_clipped = im ./ p;

% clip to 0, 1
im_01 = im_01_to_be_clipped; 
im_01(im_01 > 1) = 1; 
imshow(im_01)

% per gli esperimenti con i nostri modelli e SAR-BM3D e FANS
path_im_01 = fullfile(base_path,'mat_files/',strcat(base_name, '_im_01.mat'));

% per gli altri esperimenti
path_im_01_png = fullfile(base_path, strcat(base_name, '_im_01.png'));

save(path_im_01, 'im_01');
imwrite(im_01, path_im_01_png);



%% Single Look Complex Images

base_path = '.../dataset_SLC/TerraSAR-X/mat/';
filename = 'IMAGE_HH_SRA_spot_021.mat';

base_name = 'tsx_spotlight_021_im_01'; % for saving

mat = imread(fullfile(base_path,filename)); % (H, W, 2)

real_img = double(mat(:, :, 1));
imag_img = double(mat(:, :, 2));

img_complex = real_img + 1j*imag_img;

amplitude_img = abs(img_complex);
intensity_img = amplitude_img .^2;

% faccio l'immagine in ampiezza normalizzata al 99th percentile

p = prctile(amplitude_img(:), 99); 
ai_clip = amplitude_img ./ p;

% clip to 0, 1
amplitude_to_show = ai_clip; 
amplitude_to_show(amplitude_to_show > 1) = 1; 
% imshow(amplitude_to_show, [])

% trovo il crop con imcrop (salvo le coordinate) poi lo uso per tagliare
% tutto J = imcrop(amplitude_to_show)

% crop = [4526.5 14919.5 511 511];
real = imcrop(real_img, crop);
imag = imcrop(imag_img, crop);
amplitude = imcrop(amplitude_img, crop);
intensity = imcrop(intensity_img, crop);

% normalizzo amplitude (assumo di partire dal crop)
perc = prctile(amplitude(:), 99); 
clip = amplitude ./ perc;
im_01 = clip; 
im_01(im_01 > 1) = 1;

% per gli esperimenti con i nostri modelli e SAR-BM3D e FANS
path_im_01 = fullfile(base_path,'mat_files/',strcat(base_name, '_im_01.mat'));

% per gli altri esperimenti
path_data = fullfile(base_path, strcat(base_name, '_data.mat'));
path_im_01_png = fullfile(base_path, strcat(base_name, '_im_01.png'));

save(path_im_01, 'im_01');
save(path_data, 'amplitude', 'intensity', 'real', 'imag');
imwrite(im_01, path_im_01_png);