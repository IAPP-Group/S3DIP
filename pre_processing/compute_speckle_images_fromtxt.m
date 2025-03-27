% quello che fa compute_speckle_images.m ma avendo come input un file txt
% in cui sono salvati i path delle immagini da  processare (e gli indici)

output_base_dir = '.../L1/';

file_path = 'D:/DATASET_SAR/selected_images.txt';

fi = fopen(file_path);
out = textscan(fi, '%s %d'); % mi ritorna path e indici come cell array

fclose(fi);
list_paths = out{1, 1};
%L = 4;
L = 1;

for k=1:numel(list_paths)
    curr_path = list_paths{k}; % path immagine .tif
    fprintf("Image: %s \n", curr_path);
    spl = split(curr_path, '/');
    
    fold = spl{length(spl) - 1}; % classe del dataset in cui è contenuta l'immagine

    spli = split(spl{length(spl)}, '.'); % spli(1) è il nome dell'immagine senza ext
    
    % base path png, mat_files
    png_path = fullfile(output_base_dir, 'png', fold);
    if ~exist(png_path, 'dir')
        fprintf("creo png path: %s \n", png_path);
        mkdir(png_path)
    end
    
    matfiles_path = fullfile(output_base_dir, 'mat_files', fold);
    if ~exist(matfiles_path, 'dir')
        fprintf("creo mat files path: %s \n", matfiles_path);
        mkdir(matfiles_path)
    end
    
    % path png clean e noisy da salvare 
    pathToFinal_clean = fullfile(png_path, strcat(spli(1), '.png'));
    fprintf("pathToFinal_clean: %s \n", pathToFinal_clean{1});
    
    pathToFinal_noisy = fullfile(png_path, strcat(spli(1), '_speckle.png'));
    fprintf("pathToFinal_noisy: %s \n", pathToFinal_noisy{1});
    
    % path mat file immagine da salvare (contiene clean e noisy)
    path_mat = fullfile(matfiles_path, strcat(spli(1), '_01.mat'));
    fprintf("path_mat: %s \n", path_mat{1});
    
    if exist(path_mat{1}, 'file') == 2
        disp("Mat trovato");
    else
        disp("Processo l'immagine"); 
        % process image
        im = double(imread(curr_path)); 
        [hei, wid, ch] = size(im);
        im_01 = im ./ 255; % o im / 255
        im_01 = im2gray(im_01);  % grayscale se serve 
        U = gamrnd(L, 1/L, [hei wid]);
        % speckle image in amplitude
        imn_01 = (im_01) .* sqrt(U);

        % salvo i mat
        save(path_mat{1}, 'im_01', 'imn_01')
        clear 'im_01' 'imn_01'; 
    end

    if exist(pathToFinal_clean{1}, 'file') == 2
        disp("PNG clean trovato");
    else
        if exist('im_01','var') == 0
            % im_01 non è sul workspace
            load(path_mat{1}, 'im_01')
        end
        disp("Salvo il png clean"); 
        imwrite(im_01, pathToFinal_clean{1}); 
        clear 'im_01'; 
    end

    if exist(pathToFinal_noisy{1}, 'file') == 2
        disp("PNG noisy trovato");
    else
        if exist('imn_01','var') == 0
            % imn_01 non è sul workspace
            load(path_mat{1}, 'imn_01')
        end
        disp("Salvo il png noisy"); 
        imwrite(imn_01, pathToFinal_noisy{1}); 
        clear 'imn_01'; 
    end
end



