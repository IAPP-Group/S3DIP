function structMethods = compute_despeckled_images_from_mat()

folder_name = '.../L1/mat_files'; % input folder 
destination_folder = '.../L1'; % dest folder mat
struct_folder = '.../L1'; % dest folder struct risultati
pathToStruct = fullfile(struct_folder, 'struct_methods_L1.mat');

% folder_name = 'D:/DATASET_SAR/dataset_SLC/COSMO-SkyMed/mat/mat_files'; % input folder
% destination_folder = 'D:/DATASET_SAR/dataset_SLC/COSMO-SkyMed/mat'; % dest folder mat
% struct_folder = 'D:/DATASET_SAR/dataset_SLC/COSMO-SkyMed/mat'; % dest folder struct risultati
% pathToStruct = fullfile(struct_folder, 'struct_methods_cskm.mat');

c = dir(folder_name); 
% L = 4; % number of looks
L = 1; % number of looks
structMethods = ([]); 
s = 1; % sta su structMethods

% ora quando apro mat_files non ho direttamente le immagini ma ho le
% cartelle delle classi in cui devo entrare


for k=1:numel(c) % ciclo sulle cartelle
    if c(k).name ~= "." && c(k).name ~= ".."
        fprintf("Subfolder: %s \n", c(k).name);
        
        path_subfolder = fullfile(folder_name, c(k).name);
        ps = dir(path_subfolder);
        
        for t=1:numel(ps) % ciclo sulle immagini della singola cartella
%             fprintf("Image: %s \n", ps(t).name);
            
            if ps(t).isdir == 0
                % mat
                path_mat = fullfile(path_subfolder, ps(t).name);
                fprintf("path mat: %s \n", path_mat);
                
                load(path_mat, 'im_01', 'imn_01')
                spl = split(ps(t).name, '.');
                structMethods(s).name = path_mat;

                clean = im_01;
                noisy = imn_01; 

                %%%%%%%%%%%%%%%%%%%%%%%%% apply SARBM3D %%%%%%%%%%%%%%%%%%%%%%%%% 
                structMethods(s).sarbm3d = compute_sarbm3d(noisy, clean, L, destination_folder, spl{1}, c(k).name); 

                %%%%%%%%%%%%%%%%%%%%%%%%% apply FANS %%%%%%%%%%%%%%%%%%%%%%%%%
                structMethods(s).fans = compute_fans(noisy, clean, L, destination_folder, spl{1}, c(k).name); 

                % salvo la struct alla fine (nella cartella dove ho i metodi)
                save(pathToStruct, 'structMethods', '-v7.3', '-nocompression');

                s = s + 1; 
            end

        end

    end
end


function sarbm3d_struct = compute_sarbm3d(noisy, clean, L, destination_folder, name_im, sub_fold)
method = 'SAR-BM3D';
fprintf("Method: %s \n", method);
sarbm3d_struct = ([]); 
p = 1;

tic;
Y_sarbm3d = SARBM3D_v10(noisy,L);
elapsed_time_sarbm3d = toc;

sarbm3d_struct(p).method = method;
% structMethods(s).despeckled_image = Y_sarbm3d;
sarbm3d_struct(p).computation_time = elapsed_time_sarbm3d;

% compute psnr e ssim
% sarbm3d_struct(p).psnr = psnr(Y_sarbm3d, clean);
ps = 10 * log(1^2 / mean(mean((clean - Y_sarbm3d).^2)))/ log(10);
sarbm3d_struct(p).psnr = ps;
sarbm3d_struct(p).ssim = ssim(Y_sarbm3d, clean, 'DynamicRange', 2);

% save image despeckled as png and mat file
path_method = fullfile(destination_folder, method);

if ~exist(path_method, 'dir')
    fprintf("creo path method: %s \n", path_method);
    mkdir(path_method)
end

path_method_png = fullfile(path_method, 'png', sub_fold);
if ~exist(path_method_png, 'dir')
    fprintf("creo path method png: %s \n", path_method_png);
    mkdir(path_method_png)
end

path_method_subfold = fullfile(path_method, 'mat_files', sub_fold);
if ~exist(path_method_subfold, 'dir')
    fprintf("creo path method subfold: %s \n", path_method_subfold);
    mkdir(path_method_subfold)
end

path_mat = fullfile(path_method_subfold, strcat(name_im, '_', lower(method),'.mat'));
path_png = fullfile(path_method_png, strcat(name_im, '_', lower(method),'.png'));
disp(path_mat);
disp(path_png);
% ho già l'immagine tra 0 e 1
save(path_mat, 'Y_sarbm3d')
imwrite(Y_sarbm3d, path_png); 

% im_01 = Y_sarbm3d ./ 255; 
% save(path_mat, 'im_01')
% imwrite(im_01, path_png); 


function fans_struct = compute_fans(noisy, clean, L, destination_folder, name_im, sub_fold)
method = 'FANS';
fprintf("Method: %s \n", method);
fans_struct = ([]); 
p = 1;

tic;
Y_fans = FANS(noisy,L);
elapsed_time_fans = toc;

fans_struct(p).method = method;
fans_struct(p).computation_time = elapsed_time_fans;

% compute psnr e ssim
% fans_struct(p).psnr = psnr(Y_fans, clean);
ps = 10 * log(1^2 / mean(mean((clean - Y_fans).^2)))/ log(10);
fans_struct(p).psnr = ps;
% fans_struct(p).ssim = ssim(Y_fans, clean);
fans_struct(p).ssim = ssim(Y_fans, clean, 'DynamicRange', 2);

% % save image despeckled as png and mat file
% path_mat = fullfile(destination_folder, method, strcat(name_im, '_', lower(method),'.mat'));
% path_png = fullfile(destination_folder, method, 'png', strcat(name_im, '_', lower(method),'.png'));
% disp(path_mat);
% disp(path_png);

% save image despeckled as png and mat file
path_method = fullfile(destination_folder, method);

if ~exist(path_method, 'dir')
    fprintf("creo path method: %s \n", path_method);
    mkdir(path_method)
end

path_method_png = fullfile(path_method, 'png', sub_fold);
if ~exist(path_method_png, 'dir')
    fprintf("creo path method png: %s \n", path_method_png);
    mkdir(path_method_png)
end

path_method_subfold = fullfile(path_method, 'mat_files', sub_fold);
if ~exist(path_method_subfold, 'dir')
    fprintf("creo path method subfold: %s \n", path_method_subfold);
    mkdir(path_method_subfold)
end

path_mat = fullfile(path_method_subfold, strcat(name_im, '_', lower(method),'.mat'));
path_png = fullfile(path_method_png, strcat(name_im, '_', lower(method),'.png'));
disp(path_mat);
disp(path_png);

save(path_mat, 'Y_fans')
imwrite(Y_fans, path_png); 
% im_01 = Y_fans ./ 255; 
% save(path_mat, 'im_01')
% imwrite(im_01, path_png); 