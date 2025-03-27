""" DIP """

from __future__ import print_function

from comet_ml import Experiment, OfflineExperiment

import matplotlib.pyplot as plt
import numpy as np
import torch
from models import get_net
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from utils_mat.common_utils_mat import get_params, get_noise, np_to_torch, torch_to_np, np_to_visual
import argparse
import pickle
import scipy.io
import os
from functions import write_json
import time
from stopping_criterion import EarlyStop, myMetric

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Image Prior with ES-WMV")

    parser.add_argument("--pkl", dest="input", default=None,
                        help="path to pickle file")
    parser.add_argument("--base_save_path", dest="base_save_path", default=None,
                        help="path to the base folder for results. here we save the new folder for current experiment")
    parser.add_argument("--base_dataset_path", dest="base_dataset_path",
                        default=None,
                        help="path to the base dataset folder. here we have pckl_files, SAR_BM3D and FANS folders")
    parser.add_argument("--iter", dest="num_iter", default=3,  # None
                        help="number of iteration")
    parser.add_argument("--plot", dest="plot", default=None,  # None
                        help="True if we want to plot current output of the net. None otherwise")
    parser.add_argument("--plot_step", dest="plot_step", default=1, help="Step for plots/save images during training")

    parser.add_argument("--lr", dest="lr", default=0.01, help="Learning rate")

    # comet parameters
    parser.add_argument("--comet", dest="comet", default=0, help="1 for comet ON, 0 for comet OFF")
    parser.add_argument("--name_proj", dest="name_proj", default='despeckling_paper',
                        help="define comet ml project folder")
    parser.add_argument("--name_exp", dest="name_exp", default='we', help="name of comet ml experiment")
    parser.add_argument("--comments", dest="comments", default=None, help="comments (str) about the experiment")

    # gradient clipping
    parser.add_argument("--gc", dest="gc", default=0, help="1 to include gradient clipping (by value), 0 otherwise")

    # skip hyperparams
    # 128, 128, 4, 5, LeakyReLU
    parser.add_argument("--num_channels_out", dest="num_channels_out", default=1,
                        help="number of channels output image")
    parser.add_argument("--input_depth", dest="input_depth", default=8, help="input depth of the image tensor")
    parser.add_argument("--skip_n33d", dest="skip_n33d", default=64, help="num channels down")
    parser.add_argument("--skip_n33u", dest="skip_n33u", default=64, help="num channels up")
    parser.add_argument("--skip_n11", dest="skip_n11", default=2, help="num channels skip")
    parser.add_argument("--num_scales", dest="num_scales", default=3, help="num scales")
    parser.add_argument("--act_fun", dest="act_fun", default='ReLu', help="activation function")  # LeakyReLU

    # parser.add_argument("--device", dest="device", default='0', help="gpu device number")

    # ES-WMV
    parser.add_argument("--buffer_size", dest="buffer_size", default=100, help="buffer size for ES-WMV")  # 100
    parser.add_argument("--patience", dest="patience", default=1000, help="patience threshold for ES-WMV")  # 1000

    # image despeckled with another method
    parser.add_argument("--des_im", dest="des_im", default=None, help="image despeckled with another method")

    # loss fusion
    parser.add_argument("--loss_fusion", dest="loss_fusion", default=None,
                        help="if 1 it takes the name of the image under exam to recover the despeckled versions of it, "
                             "using the mean of sarbm3d and fans methods in the total loss computation")

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # dtype = torch.cuda.FloatTensor

    print("call parameters: ", args.input, args.name_exp, args.num_iter, args.plot_step, args.base_save_path, args.des_im, args.loss_fusion)

    ######################## stooping criterion
    buffer_size = int(args.buffer_size)
    patience = int(args.patience)
    variance_history = []
    x_axis = []
    earlystop = EarlyStop(size=buffer_size, patience=patience)
    ########################

    # Save path
    save_path = os.path.join(args.base_save_path.rstrip('\r'), args.name_exp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("save path: ", save_path)

    # if torch.cuda.is_available():
    #     device = torch.device('cuda:' + args.device)
    #     print('Using device:', args.device, torch.cuda.get_device_name(int(args.device)))
    # else:
    #     device = 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    torch.autograd.set_detect_anomaly(True)  # add

    # Setup (some static params from the original implementation of DIP)
    imsize = -1
    PLOT = args.plot
    plot_step = int(args.plot_step)
    # sigma = 25
    # sigma_ = sigma / 255.

    INPUT = 'noise'  # 'meshgrid'
    pad = 'reflection'
    OPT_OVER = 'net'  # 'net,input'

    reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50
    LR = float(args.lr)

    OPTIMIZER = 'adam'  # 'LBFGS'
    exp_weight = 0.99

    nco = int(args.num_channels_out)  # 1
    input_depth = int(args.input_depth)  # 32
    num_iter = int(args.num_iter)  # numero di epoche

    # Load images
    # args.input è un file pickle su cui ho salvato la matrice dell'immagine rumorosa e dell'immagine pulita
    f = open(args.input, 'rb')
    noisy_image, clean_image = pickle.load(f)  # nd.arrays (H, W)

    # noisy image
    # img_noisy, h_pad_noisy, w_pad_noisy = padding(noisy_image, d=input_depth)
    img_noisy = noisy_image[None, :]
    img_noisy = img_noisy.astype(np.float32)  # nd.array (C, H, W)
    img_noisy_torch = np_to_torch(img_noisy).to(device)  # type(dtype)
    print("img noisy torch shape: ", img_noisy_torch.shape)

    # clean image
    # img, h_pad, w_pad = padding(clean_image, d=input_depth)
    img = clean_image[None, :]  # 1, H, W
    img = img.astype(np.float32)  # nd.array (C, H, W)
    img_torch = np_to_torch(img).to(device)

    if args.des_im is not None:
        f = open(args.des_im.rstrip('\r'), 'rb')
        # './data/SAR/pickle/aereal_dataset/fans/building05_fans.pckl'
        des_im = pickle.load(f)  # nd.array nel range (0,1)
        des_im = np_to_torch(des_im.astype(np.float32)).to(device)
        f.close()
        print("des im shape: ", des_im.shape)

    ## fusione nella loss (faccio la media tra sar-bm3d e fans)
    loss_fusion = args.loss_fusion
    if args.loss_fusion is not None:
        spl = args.input.split('/')[-1]
        loss_fusion = spl.split('.')[0]  # [:-3]
        print('loss fusion: ', loss_fusion)

        # fname_s = 'D:/DATASET_SAR/SAR-BM3D/pckl_files/' + loss_fusion[:-5] + '/' + loss_fusion + '_sar-bm3d.pckl'
        fname_s = args.base_dataset_path + '/SAR-BM3D/pckl_files/' + loss_fusion[
                                                                     :-5] + '/' + loss_fusion + '_sar-bm3d.pckl'
        f = open(fname_s.rstrip('\r'), 'rb')
        des_im_s = pickle.load(f)  # nd.array nel range (0,1)
        des_im_s = np_to_torch(des_im_s.astype(np.float32))
        f.close()

        # fname_f = 'D:/DATASET_SAR/FANS/pckl_files/' + loss_fusion[:-5] + '/' + loss_fusion + '_fans.pckl'  # local
        fname_f = args.base_dataset_path + '/FANS/pckl_files/' + loss_fusion[:-5] + '/' + loss_fusion + '_fans.pckl'
        f = open(fname_f.rstrip('\r'), 'rb')
        des_im_f = pickle.load(f)  # nd.array nel range (0,1)
        des_im_f = np_to_torch(des_im_f.astype(np.float32))
        f.close()

        des_im_fus = ((des_im_s + des_im_f) / 2).to(device)
        # print("des im fus shape: ", des_im_fus.shape)

    # Z net input (not under the computational graph)
    net_input = get_noise(input_depth, INPUT, (img.shape[1], img.shape[2])).to(device).detach()
    print("net input shape: ", net_input.shape)

    # NET
    net = get_net(input_depth, 'skip', pad, n_channels=nco,  # ho un'immagine in bianco e nero
                  skip_n33d=int(args.skip_n33d),
                  skip_n33u=int(args.skip_n33u),
                  skip_n11=int(args.skip_n11),
                  num_scales=int(args.num_scales),
                  upsample_mode='bilinear',
                  act_fun=str(args.act_fun)).to(device)  # type(dtype)

    # Compute number of NET parameters
    s = sum([np.prod(list(p.size())) for p in net.parameters()])
    print('Number of NET params: %d' % s)

    # Optimization
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None

    p = get_params(OPT_OVER, net, net_input, noise_tensor=None, downsampler=None)  # total params to optimize
    s = sum([np.prod(list(g.size())) for g in p])
    print('Number of TOTAL params: %d' % s)

    # optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    optimizer = torch.optim.Adam(p, lr=LR)

    # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=40, threshold=0.0001,
    #                                                    threshold_mode='rel',
    #                                                    cooldown=0, min_lr=0, eps=1e-08, verbose=False)

    # Loss(es)
    mse = torch.nn.MSELoss().to(device)  # type(dtype)

    # # ES
    # buffer_size = int(args.buffer_size)
    # patience = int(args.patience)

    # COMET
    experiment = None
    if int(args.comet) == 0:
        # Comet ml integration
        experiment = OfflineExperiment(offline_directory=save_path + '/COMET_OFFLINE',project_name=args.name_proj)
    else:
        # matplotlib.use('TkAgg')
        experiment = Experiment(project_name=args.name_proj)

    experiment.set_name(args.name_exp)
    ek = experiment.get_key()

    hyper_params = {
        "num_epochs": num_iter,
        "learning_rate": LR,
        "input_depth": input_depth,
        "skip_n33d": int(args.skip_n33d),
        "skip_n33u": int(args.skip_n33u),
        "skip_n11": int(args.skip_n11),
        "num_scales": int(args.num_scales),
        "act_fun": str(args.act_fun),
        "gradient_clipping": int(args.gc),
        'buffer_size': buffer_size,
        'patience': patience,
        'opt_over': OPT_OVER,
        'des_im': args.des_im,
        'loss_fusion': loss_fusion
    }
    experiment.log_parameters(hyper_params)
    experiment.set_model_graph(net)
    experiment.log_other('num_parameters', s)
    if args.comments:
        experiment.log_other('comments', args.comments)

    # # Additional Info when using cuda
    # if device.type == 'cuda':
    #     print(torch.cuda.get_device_name(0))

    i = 0
    vi_ref = {}
    vi_ref_sm = {}
    ssim_ref = {}
    ssim_ref_sm = {}
    log_data = []

    loss_history = []
    s_loss_history = []
    des_loss_history = []
    ssim_history = []
    psnr_history = []

    # for smoothed image
    psnr_history_avg = []
    ssim_history_avg = []

    bo_loss = None

    start = time.time()

    #################################################### Training loop
    for i in range(num_iter):

        optimizer.zero_grad()

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)  # ...

        # Compute prediction
        out = net(net_input)

        # 'Smoothing'
        if out_avg is None:  # all'iterazione 0 non c'è
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
            # nuovo valore = valore corrente * peso + valore new * (1 - peso)

        # img ratio

        # Loss(es)
        spatial_loss = mse(out, img_noisy_torch)
        if args.des_im is not None:
            bo_loss = mse(out, des_im[None, :])
            total_loss = spatial_loss + bo_loss

        elif loss_fusion is not None:
            bo_loss = mse(out, des_im_fus[None, :])
            total_loss = spatial_loss + bo_loss

        else:
            total_loss = spatial_loss

        if int(args.comet) is not None:
            experiment.log_metric('Total_loss', total_loss.item(), i)

            if bo_loss is not None:
                experiment.log_metric('Des_loss', bo_loss.item(), i)
                experiment.log_metric('Spatial_loss', spatial_loss.item(), i)

        # Backward
        total_loss.backward()

        if bo_loss is not None:
            des_loss_history.append(bo_loss.item())
            s_loss_history.append(spatial_loss.item())

        loss_history.append(total_loss.item())

        if int(args.gc) == 1:
            # par = get_params(OPT_OVER, net, net_input, spt, downsampler=None)  # devo ricalcolarli ogni volta ?
            torch.nn.utils.clip_grad_value_(p, clip_value=1.0)
            # nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)

        # torch_to_np(img_var) --> img_var.detach().cpu().numpy()[0]
        out_depad = torch_to_np(out)  # (C, H, W)
        out_avg_depad = torch_to_np(out_avg)

        # TODO: PER POTER FARE CONFRONTI POTREI CALCOLARE L'ISTOGRAMMA DELLO SPECKLE OTTENUTO COME IMMAGINE RAPPORTO
        # appesantisco un po' il codice .. se potessi calcolarlo a posteriori ... potrei salvare uno zip di queste immaginiin pckl
        #  speckle_tensor_detached = speckle_tensor.detach().cpu().numpy()

        # Compute PSNR
        psnr_gt = peak_signal_noise_ratio(img, out_depad)  # data_range=1.
        psnr_gt_sm = peak_signal_noise_ratio(img, out_avg_depad)

        # Compute SSIM
        ssim_gt = structural_similarity(img[0], out_depad[0])  # data_range=dr_clean
        ssim_gt_sm = structural_similarity(img[0], out_avg_depad[0])  # data_range=dr_clean

        psnr_history.append(psnr_gt)
        ssim_history.append(ssim_gt)

        psnr_history_avg.append(psnr_gt_sm)
        ssim_history_avg.append(ssim_gt_sm)

        # stopping criterion
        # variance history
        out_depad_flatten = out_depad.reshape(-1)  # flatten
        earlystop.update_img_collection(out_depad_flatten)
        img_collection = earlystop.get_img_collection()

        if len(img_collection) == buffer_size:
            ave_img = np.mean(img_collection, axis=0)
            variance = []
            for tmp in img_collection:
                variance.append(myMetric(ave_img, tmp))
            cur_var = np.mean(variance)
            cur_epoch = i
            variance_history.append(cur_var)
            x_axis.append(cur_epoch)
            if earlystop.stop == False:
                earlystop.stop, best_updated = earlystop.check_stop(cur_var, cur_epoch)
                # qui aggiorna i valori best - non ho bisogno di fare altri confronti, mi basta aggiornare l'immagine che salvo
                if best_updated:
                    es_ref = {'out': out_depad, 'out_avg': out_avg_depad,
                              'psnr_gt': psnr_history[earlystop.best_epoch],
                              'psnr_gt_sm': psnr_history_avg[earlystop.best_epoch],
                              'ssim_gt': ssim_history[earlystop.best_epoch],
                              'ssim_gt_sm': ssim_history_avg[earlystop.best_epoch],
                              'it': str(earlystop.best_epoch)}

        #####

        if int(args.comet) is not None:
            experiment.log_metric('Psnr_gt', psnr_gt, i)
            experiment.log_metric('Psnr_gt_sm', psnr_gt_sm, i)
            experiment.log_metric('Ssim_gt', ssim_gt, i)
            experiment.log_metric('Ssim_gt_sm', ssim_gt_sm, i)

        print('Iteration %05d   Loss %f   PSNR_gt: %f  PSNR_gt_sm: %f' % (
            i, total_loss.item(), psnr_gt, psnr_gt_sm), '\r', end='')

        if i == 0:
            vi_ref = {'out': out_depad, 'out_avg': out_avg_depad,
                      'psnr_gt': psnr_gt, 'psnr_gt_sm': psnr_gt_sm, 'ssim_gt': ssim_gt, 'ssim_gt_sm': ssim_gt_sm,
                      'it': str(i)}
            vi_ref_sm = ssim_ref = ssim_ref_sm = vi_ref  # same initialization for all  # ssim_ref_sm

        if psnr_gt > vi_ref['psnr_gt']:
            # print("Out image: ", str(i), " has a greater value of PSNR GT --> this is the new reference value")
            vi_ref = {'out': out_depad, 'out_avg': out_avg_depad,
                      'psnr_gt': psnr_gt, 'psnr_gt_sm': psnr_gt_sm, 'ssim_gt': ssim_gt, 'ssim_gt_sm': ssim_gt_sm,
                      'it': str(i)}

        # do the same also for smoothed version of the out image
        if psnr_gt_sm > vi_ref_sm['psnr_gt_sm']:
            # print("Out image sm: ", str(i), " has a greater value of PSNR GT SM --> this is the new reference value")
            vi_ref_sm = {'out': out_depad, 'out_avg': out_avg_depad,
                         'psnr_gt': psnr_gt, 'psnr_gt_sm': psnr_gt_sm, 'ssim_gt': ssim_gt, 'ssim_gt_sm': ssim_gt_sm,
                         'it': str(i)}

        if ssim_gt > ssim_ref['ssim_gt']:
            # print("Out image: ", str(i), " has a greater value of SSIM GT --> this is the new reference value")
            ssim_ref = {'out': out_depad, 'out_avg': out_avg_depad,
                        'psnr_gt': psnr_gt, 'psnr_gt_sm': psnr_gt_sm, 'ssim_gt': ssim_gt, 'ssim_gt_sm': ssim_gt_sm,
                        'it': str(i)}

        if ssim_gt_sm > ssim_ref['ssim_gt_sm']:
            # print("Out image: ", str(i), " has a greater value of SSIM GT SM --> this is the new reference value")
            ssim_ref_sm = {'out': out_depad, 'out_avg': out_avg_depad,
                           'psnr_gt': psnr_gt, 'psnr_gt_sm': psnr_gt_sm, 'ssim_gt': ssim_gt, 'ssim_gt_sm': ssim_gt_sm,
                           'it': str(i)}

        if i == 0 or i % plot_step == 0 or i == (num_iter - 1):
            # todo: qui funziona se tutte le immagini hanno le stesse dimensioni ... se faccio padding effettivamente,
            #  servirà depaddare anche lo speckle ...

            # # Save image
            img_saved = np_to_visual(out_depad)  # (H, W)
            # np.clip(img_saved, 0, 1)
            plt.imsave(save_path + '/it' + str(i) + '.png', img_saved, cmap='gray')
            # imsave salva le immagini tra min e max dell'immagine, di default se metto cmap='gray'

            # img_noisy_2d = np_to_visual(img_noisy)  # H, W
            # appr_sp = img_noisy_2d / img_saved
            #
            # plt.imsave(save_path + '/appr_sp' + str(i) + '.png', appr_sp, cmap='gray')

            # backtracking (?)

        # Update weights
        optimizer.step()

        # sched step
        # sched.step(total_loss)

        i += 1

    end = time.time()
    print("Execution time: ", end - start)

    experiment.end()

    log_data.append({'execution_time': end - start})
    log_data.append({'experiment_key': ek})

    ## save psnr, psrn_sm arrays
    f = open(save_path + '/psnr_list.pckl', 'wb')
    pickle.dump(psnr_history, f)
    f.close()

    f = open(save_path + '/psnr_list_sm.pckl', 'wb')
    pickle.dump(psnr_history_avg, f)
    f.close()

    f = open(save_path + '/ssim_list.pckl', 'wb')
    pickle.dump(ssim_history, f)
    f.close()

    f = open(save_path + '/ssim_list_sm.pckl', 'wb')
    pickle.dump(ssim_history_avg, f)
    f.close()

    if bo_loss is not None:
        f = open(save_path + '/des_loss_list.pckl', 'wb')  # regularization loss
        pickle.dump(des_loss_history, f)
        f.close()

        f = open(save_path + '/s_loss_list.pckl', 'wb')  # spatial loss
        pickle.dump(s_loss_history, f)
        f.close()

    f = open(save_path + '/loss_list.pckl', 'wb')
    pickle.dump(loss_history, f)  # total loss
    f.close()
    ##

    f = open(save_path + '/variance_history.pckl', 'wb')
    pickle.dump([x_axis, variance_history], f)
    f.close()

    # Save
    img_saved = np_to_visual(vi_ref['out'])
    img_saved_sm = np_to_visual(vi_ref['out_avg'])

    # Save as mat and pickle
    scipy.io.savemat(save_path + '/images_max_psnr.mat', mdict={'img_saved': img_saved, 'img_saved_sm': img_saved_sm})
    # f = open(save_path + '/images_max_psnr.pckl', 'wb')
    # pickle.dump([img_saved, img_saved_sm], f)
    # f.close()

    # Save as images
    # np.clip(img_saved, 0, 1)
    plt.imsave(save_path + '/img_max_it' + vi_ref['it'] + '.png', img_saved, cmap='gray')
    plt.imsave(save_path + '/img_max_sm_it' + vi_ref['it'] + '.png', img_saved_sm, cmap='gray')

    # if vi_ref['it'] != vi_ref_sm['it'] or vi_ref['it'] != ssim_ref['it']:
    if vi_ref['it'] != vi_ref_sm['it'] or vi_ref['it'] != ssim_ref['it'] or vi_ref['it'] != ssim_ref_sm['it']:

        if vi_ref['it'] != vi_ref_sm['it']:
            ## save also the pckl of the max epoch for smoothed version of the out image ...
            img_saved = np_to_visual(vi_ref_sm['out'])
            img_saved_sm = np_to_visual(vi_ref_sm['out_avg'])
            scipy.io.savemat(save_path + '/images_max_psnr_smooth' + vi_ref_sm['it'] + '.mat',
                             mdict={'img_saved': img_saved, 'img_saved_sm': img_saved_sm})
            # f = open(save_path + '/images_max_psnr_smooth' + vi_ref_sm['it'] + '.pckl', 'wb')
            # pickle.dump([img_saved, img_saved_sm], f)
            # f.close()
            ##

        if vi_ref['it'] != ssim_ref['it']:
            ## save also the pckl of the max epoch for ssim of the out image ...
            img_saved = np_to_visual(ssim_ref['out'])
            img_saved_sm = np_to_visual(ssim_ref['out_avg'])
            scipy.io.savemat(save_path + '/images_max_ssim' + ssim_ref['it'] + '.mat',
                             mdict={'img_saved': img_saved, 'img_saved_sm': img_saved_sm})
            # f = open(save_path + '/images_max_ssim' + ssim_ref['it'] + '.pckl', 'wb')
            # pickle.dump([img_saved, img_saved_sm], f)
            # f.close()
            ##

        if vi_ref['it'] != ssim_ref_sm['it']:
            ## save also the pckl of the max epoch for ssim smoothed of the out image ...
            img_saved = np_to_visual(ssim_ref_sm['out'])
            img_saved_sm = np_to_visual(ssim_ref_sm['out_avg'])
            scipy.io.savemat(save_path + '/images_max_ssim_smooth' + ssim_ref_sm['it'] + '.mat',
                             mdict={'img_saved': img_saved, 'img_saved_sm': img_saved_sm})
            # f = open(save_path + '/images_max_ssim_smooth' + ssim_ref_sm['it'] + '.pckl', 'wb')
            # pickle.dump([img_saved, img_saved_sm], f)
            # f.close()
            ##

    if best_updated:
        # Save ES-WMV images
        img_es_saved = np_to_visual(es_ref['out'])
        img_es_sm_saved = np_to_visual(es_ref['out_avg'])

        scipy.io.savemat(save_path + '/images_max_es_psnr.mat', mdict={'img_es_saved': img_es_saved,
                                                                    'img_es_sm_saved': img_es_sm_saved})

        plt.imsave(save_path + '/img_es_max_it' + es_ref['it'] + '.png', img_es_saved, cmap='gray')
        plt.imsave(save_path + '/img_es_max_sm_it' + es_ref['it'] + '.png', img_es_sm_saved, cmap='gray')

    # log
    log_data.append({'best_epoch': vi_ref['it'], 'psnr_gt': vi_ref['psnr_gt'], 'psnr_gt_sm': vi_ref['psnr_gt_sm'],
                     'ssim_gt': vi_ref['ssim_gt'], 'ssim_gt:sm': vi_ref['ssim_gt_sm']})

    log_data.append({'best_epoch_sm': vi_ref_sm['it'], 'psnr_gt': vi_ref_sm['psnr_gt'],
                     'psnr_gt_sm': vi_ref_sm['psnr_gt_sm'],
                     'ssim_gt': vi_ref_sm['ssim_gt'], 'ssim_gt:sm': vi_ref_sm['ssim_gt_sm']})

    log_data.append({'best_epoch_ssim': ssim_ref['it'], 'psnr_gt': ssim_ref['psnr_gt'],
                     'psnr_gt_sm': ssim_ref['psnr_gt_sm'],
                     'ssim_gt': ssim_ref['ssim_gt'], 'ssim_gt:sm': ssim_ref['ssim_gt_sm']})

    log_data.append({'best_epoch_ssim_sm': ssim_ref_sm['it'], 'psnr_gt': ssim_ref_sm['psnr_gt'],
                     'psnr_gt_sm': ssim_ref_sm['psnr_gt_sm'],
                     'ssim_gt': ssim_ref_sm['ssim_gt'], 'ssim_gt:sm': ssim_ref_sm['ssim_gt_sm']})

    if best_updated:
        # ES-WMV detection gaps
        # psnr
        psnr_es = psnr_history[earlystop.best_epoch]  # valore psnr alla 'best epoch' rilevata con ES-WMV
        psnr_max = max(psnr_history)  # data[1]['psnr_gt']

        # ssim
        ssim_es = ssim_history[earlystop.best_epoch]  # valore ssim alla 'best epoch' rilevata con ES-WMV
        ssim_max = max(ssim_history)  # data[1]['ssim_gt']

        detection_psnr_gap = psnr_max - psnr_es
        detection_ssim_gap = ssim_max - ssim_es

        log_data.append({'best_epoch_ES': es_ref['it'], 'psnr_gt': es_ref['psnr_gt'],
                        'psnr_gt_sm': es_ref['psnr_gt_sm'],
                        'ssim_gt': es_ref['ssim_gt'], 'ssim_gt_sm': es_ref['ssim_gt_sm'],
                        'detection_psnr_gap': detection_psnr_gap, 'detection_ssim_gap': detection_ssim_gap})

    write_json(log_data, save_path + '/log_file.json')

    ### plot variance and psnr trends with earlystop.best_epoch
    # fig, ax1 = plt.subplots()
    #
    # color = 'tab:red'
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('PSNR', color=color)
    # ax1.plot(psnr_history, color=color)
    # ax1.tick_params(axis='y', labelcolor=color)
    #
    # ax2 = ax1.twinx()
    #
    # color = 'tab:blue'
    # ax2.set_ylabel('Variance', color=color)
    # ax2.plot(x_axis, variance_history, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    # fig.tight_layout()
    # plt.title('ES-WMV')
    # plt.axvline(x=earlystop.best_epoch, label='detection', color='y')
    # plt.legend()
    # # plt.show()
    # plt.savefig(save_path + '/ES-WMV.png', bbox_inches='tight', dpi=300)
