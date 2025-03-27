import json
import scipy.io
import pickle
import os
import glob
import platform
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt


def convert_mat2pickle(input_folder, dest_folder):
    """ Convert the files of noisy, clean images from mat to pickle """
    
    # example
    # input_folder = 'D:/DATASET_SAR/L1/mat_files'
    # dest_folder = 'D:/DATASET_SAR/L1/pckl_files'  # crea

    subfolders = os.listdir(input_folder)
    for s in subfolders:
        subfold_path = input_folder + '/' + s
        print("subfolder path: ", subfold_path)

        mat_files = glob.glob(os.path.join(subfold_path, '*.mat'))

        for i in range(len(mat_files)):
            filename = mat_files[i].replace('\\', '/')
            print("Filename: ", filename)
            spl = filename.split('/')[-1]
            name = spl.split('.')[0]
            # dest_filename = dest_folder + '/' + s + '/' + name + '.pckl'
            dest_subfold = dest_folder + '/' + s
            if not os.path.exists(dest_subfold):
                os.makedirs(dest_subfold)

            dest_filename = dest_subfold + '/' + name + '.pckl'
            print("Destination filename: ", dest_filename)
            if not os.path.exists(dest_filename):
                mat = scipy.io.loadmat(filename)
                im_01 = mat['im_01']
                imn_01 = mat['imn_01']  # im tra 0 e 1
                f = open(dest_filename, 'wb')
                pickle.dump([imn_01, im_01], f)  # noisy, clean
                f.close()
            else:
                print("Already done")


def convert_mat2pickle_sarbm3d_fans(base_input_folder, base_dest_folder, method):

    """ Convert the denoised files of SAR-BM3D and FANS from mat to pickle """
    
    # example
    # base_input_folder = 'D:/DATASET_SAR/L1/'
    # base_dest_folder = 'D:/DATASET_SAR/L1/'  # crea
    # method can be 'SAR-BM3D' or 'FANS'

    if method == 'SAR-BM3D':
        name_var = 'Y_sarbm3d'
    else:
        # FANS
        name_var = 'Y_fans'

    input_folder = base_input_folder + method + '/mat_files'
    dest_folder = base_dest_folder + method + '/mat_files'

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    mat_files = os.listdir(input_folder)
    for i in range(len(mat_files)):

        filename = input_folder + '/' + mat_files[i].replace('\\', '/')
        print("Filename: ", filename)
        spl = filename.split('/')[-1]
        name = spl.split('.')[0]

        dest_filename = dest_folder + '/' + name + '.pckl'
        print("Destination filename: ", dest_filename)
        # save
        if not os.path.exists(dest_filename):
            mat = scipy.io.loadmat(filename)
            im_01 = mat[name_var]
            f = open(dest_filename, 'wb')
            pickle.dump(im_01, f)  # despeckled image with method
            f.close()
        else:
            print("Already done")


def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def get_psnr_ratio_fig(all_datas, num_iter, ylim=35, ylabel='', save_path='', img_name=''):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(0, num_iter)
    ax.set_ylim(0, ylim)

    plt.xlabel("Optimization Iteration")
    # plt.ylabel(ylabel)
    # plt.title(img_name)

    label_list = ['PSNR', 'Ratio']
    color_list = ['#d94a31', '#4b43db']

    rate = 1
    for i in range(len(all_datas)):
        plt.plot(range(0, num_iter, rate), all_datas[i][0:num_iter:rate], linewidth=4, color=color_list[i],
                 label=label_list[i])

    plt.legend(loc=0, )
    plt.grid()
    plt.savefig(save_path)
    # plt.show()
    plt.close()


if __name__ == '__main__':

    # convert_mat2pickle(input_folder, dest_folder)
    # convert_mat2pickle_sarbm3d_fans(base_input_folder, base_dest_folder, method)
    pass
