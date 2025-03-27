""" S3DIP for real SAR images"""

from __future__ import print_function

from comet_ml import Experiment, OfflineExperiment

import matplotlib.pyplot as plt
import numpy as np
import torch
from models import get_net
from utils_mat.common_utils_mat import get_params, get_noise, np_to_torch, torch_to_np, np_to_visual
import argparse
import pickle
import scipy.io
from hist_loss_broadcast import histogram_loss, compute_h_broadcast
import os
from functions import write_json
import time
from stopping_criterion import EarlyStop, myMetric

# nb: ho commentato in common.py della libreria l'assert sul valore più alto di data range ... in modo che calcolasse
# ssim anche se una delle due immagini assumeva valori sopra 1 ...

# init val = 0.918 (mediana di U 10.0000x10.000)
# init val = 0.958 (mediana di sqrt U 10.0000x10.000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="S3DIP applied to real SAR images.")
    parser.add_argument("--pkl", dest="input", default=None, help="path to pickle file")  
    parser.add_argument("--base_save_path", dest="base_save_path", default=None,
                        help="path to the base folder for results. here we save the new folder for current experiment")
    parser.add_argument("--base_dataset_path", dest="base_dataset_path", default=None,
                        help="path to the base dataset folder. here we have pckl_files, SAR_BM3D and FANS folders")
    parser.add_argument("--iter", dest="num_iter", default=3,  # None
                        help="number of iteration")
    parser.add_argument("--num_channels_out", dest="num_channels_out", default=1,  # 3
                        help="number of channels output image")
    parser.add_argument("--input_depth", dest="input_depth", default=8, help="3/1 for gt == 0, 32 for gt == 1")
    parser.add_argument("--plot", dest="plot", default=None,  # None
                        help="True if we want to plot current output of the net. None otherwise")
    parser.add_argument("--plot_step", dest="plot_step", default=1, help="Step for plots/save images during training")

    parser.add_argument("--h_weight", dest="h_weight", default=1, help="Histogram loss weight")
    parser.add_argument("--s_weight", dest="s_weight", default=1, help="Spatial loss weight")
    parser.add_argument("--init_val", dest="init_val", default=0.5, help="Initial value for speckle matrix")
    parser.add_argument("--lr", dest="lr", default=0.01, help="Learning rate")
    parser.add_argument("--square", dest="square", default=1, help="1 if sqrt reference speckle is passed "
                                                                   "(speckle noise --> sqrt of reference speckle, "
                                                                   "0 if original reference speckle is passed "
                                                                   "(speckle noise --> original reference speckle")
    parser.add_argument("--h_path", dest="h_path",
                        default=None,
                        help="path to the reference histogram (h_ref)")

    # comet parameters
    parser.add_argument("--comet", dest="comet", default=0, help="1 for comet ON, 0 for comet OFF")
    parser.add_argument("--name_proj", dest="name_proj", default='despeckling_paper', help="define comet ml project folder")
    parser.add_argument("--name_exp", dest="name_exp", default='we', help="name of comet ml experiment")
    parser.add_argument("--comments", dest="comments", default=None, help="comments (str) about the experiment")

    # gradient clipping
    parser.add_argument("--gc", dest="gc", default=0, help="1 to include gradient clipping (by value), 0 otherwise")

    # skip hyperparams
    parser.add_argument("--skip_n33d", dest="skip_n33d", default=64, help="num channels down")
    parser.add_argument("--skip_n33u", dest="skip_n33u", default=64, help="num channels up")
    parser.add_argument("--skip_n11", dest="skip_n11", default=2, help="num channels skip")
    parser.add_argument("--num_scales", dest="num_scales", default=3, help="num scales")
    parser.add_argument("--act_fun", dest="act_fun", default='ReLu', help="activation function")

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

    print("call parameters: ", args.input, args.name_exp, args.num_iter, args.plot_step, args.base_save_path, args.des_im, args.loss_fusion, args.base_dataset_path)

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

    if torch.cuda.is_available():
        device = torch.device('cuda:' + args.device)
        print('Using device:', args.device, torch.cuda.get_device_name(int(args.device)))
    else:
        device = 'cpu'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    torch.autograd.set_detect_anomaly(True)  # add

    # Setup (some static params from the original implementation of DIP)
    imsize = -1
    PLOT = bool(args.plot)
    plot_step = int(args.plot_step)
    # sigma = 25
    # sigma_ = sigma / 255.

    INPUT = 'noise'  # 'meshgrid'
    pad = 'reflection'
    # OPT_OVER = 'net'  # 'net,input'
    OPT_OVER = 'net,noise'

    reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50
    LR = float(args.lr)

    OPTIMIZER = 'adam'  # 'LBFGS'
    exp_weight = 0.99

    nco = int(args.num_channels_out)  # 1
    input_depth = int(args.input_depth)  # 32
    num_iter = int(args.num_iter)  # numero di epoche

    # Setup for histogram loss
    h_weight = float(args.h_weight)
    s_weight = float(args.s_weight)
    init_val = float(args.init_val)
    square = int(args.square)

    # Load images
    # in questo caso non abbiamo un'immagine 'pulita'
    f = open(args.input, 'rb')
    noisy_image = pickle.load(f)  # nd.arrays (H, W)

    # noisy image
    # img_noisy, h_pad_noisy, w_pad_noisy = padding(noisy_image, d=input_depth)
    img_noisy = noisy_image[None, :]
    img_noisy = img_noisy.astype(np.float32)  # nd.array (C, H, W)
    img_noisy_torch = np_to_torch(img_noisy).to(device)  # type(dtype)
    print("img noisy torch shape: ", img_noisy_torch.shape)

    # Speckle initialization
    speckle = (init_val * np.ones((img_noisy.shape[1], img_noisy.shape[2]))).astype(np.float32)  # nd.array (H, W)
    spt = np_to_torch(speckle).to(device)

    print("speckle tensor shape: ", spt.shape)

    if args.des_im is not None:
        f = open(args.des_im.rstrip('\r'), 'rb')
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

        n_looks = args.input.split('/')[-3]  # L1 in .../L1/pckl_files/nome.pckl

        # fname_s = args.base_dataset_path + '/' + n_looks + '/SAR-BM3D/pckl_files/' + loss_fusion + '_sar-bm3d.pckl'
        fname_s = args.base_dataset_path.rstrip('\r') + '/SAR-BM3D/pckl_files/' + loss_fusion + '_sar-bm3d.pckl'
        f = open(fname_s.rstrip('\r'), 'rb')
        des_im_s = pickle.load(f)  # nd.array nel range (0,1)
        des_im_s = np_to_torch(des_im_s.astype(np.float32))
        f.close()

        # fname_f = args.base_dataset_path + '/' + n_looks + '/FANS/pckl_files/' + loss_fusion + '_fans.pckl'
        fname_f = args.base_dataset_path.rstrip('\r') + '/FANS/pckl_files/' + loss_fusion + '_fans.pckl'
        f = open(fname_f.rstrip('\r'), 'rb')
        des_im_f = pickle.load(f)  # nd.array nel range (0,1)
        des_im_f = np_to_torch(des_im_f.astype(np.float32))
        f.close()

        des_im_fus = ((des_im_s + des_im_f) / 2).to(device)
        print("des im fus shape: ", des_im_fus.shape)

    # Z net input (not under the computational graph)
    net_input = get_noise(input_depth, INPUT, (img_noisy.shape[1], img_noisy.shape[2])).to(device).detach()
    print("net input shape: ", net_input.shape)

    # if torch.cuda.device_count() > 1:
    #     print("Use: ", torch.cuda.device_count() + ' GPUs')
    #     net = get_net(input_depth, 'skip', pad, n_channels=nco,  # ho un'immagine in bianco e nero
    #               skip_n33d=int(args.skip_n33d),
    #               skip_n33u=int(args.skip_n33u),
    #               skip_n11=int(args.skip_n11),
    #               num_scales=int(args.num_scales),
    #               upsample_mode='bilinear',
    #               act_fun=str(args.act_fun))
    #     net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[0,1])

    net = get_net(input_depth, 'skip', pad, n_channels=nco,  # ho un'immagine in bianco e nero
                skip_n33d=int(args.skip_n33d),
                skip_n33u=int(args.skip_n33u),
                skip_n11=int(args.skip_n11),
                num_scales=int(args.num_scales),
                upsample_mode='bilinear',
                act_fun=str(args.act_fun)).to(device)  # type(dtype)

    # Compute number of parameters
    mp = filter(lambda p: p.requires_grad, net.parameters())
    s = sum([np.prod(list(p.size())) for p in mp])
    print('Number of NET params: %d' % s)

    # Histogram of Nref (noise of reference for the speckle distribution)
    name = str(args.h_path)

    f = open(name.rstrip('\r'), 'rb')
    h_ref, to_tensor = pickle.load(f)
    to_tensor = to_tensor.to(device)  # torch.float64
    h_ref = h_ref.to(device)
    step = (to_tensor[0, 1] - to_tensor[0, 0]).item()
    f.close()

    softplus = torch.nn.Softplus().to(device)  # apply it on the speckle

    # Optimization
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None

    p = get_params(OPT_OVER, net, net_input, spt, downsampler=None)  # total params to optimize
    s = sum([np.prod(list(g.size())) for g in p])
    print('Number of TOTAL params: %d' % s)

    # optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    optimizer = torch.optim.Adam(p, lr=LR)

    # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=40, threshold=0.0001,
    #                                                    threshold_mode='rel',
    #                                                    cooldown=0, min_lr=0, eps=1e-08, verbose=False)

    # sched = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=LR, pct_start=0.05,
    #                                             total_steps=num_iter * 4,
    #                                             anneal_strategy='cos')

    # Loss(es)
    mse = torch.nn.MSELoss().to(device)  # type(dtype)

    experiment = None
    if int(args.comet) == 0:
        # Comet ml integration
        experiment = OfflineExperiment(offline_directory=save_path + '/COMET_OFFLINE',
                                       project_name=args.name_proj)
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
        "square": square,
        "s_weight": s_weight,
        "h_weight": h_weight,
        "init_val": init_val,
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

    i = 0
    vi_ref = {}
    h_data = []
    log_data = []
    # k = 1  # per la lista degli istogrammi

    # variables to compute the histogram of speckle tensor
    t_resh = to_tensor.reshape(-1)  # flatten
    t_resh = t_resh.reshape(-1, 1, 1)
    resh_expanded = t_resh.expand(-1, spt.shape[1], spt.shape[2])
    # print("resh_expanded shape: ", resh_expanded.shape)
    
    del t_resh
    torch.cuda.empty_cache()

    loss_history = []
    h_loss_history = []
    s_loss_history = []
    des_loss_history = []

    bo_loss = None

    start = time.time()

    #################################################### Training loop
    for i in range(num_iter):

        optimizer.zero_grad()

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)  # ...

        # Compute prediction
        out = net(net_input)

        speckle_tensor = softplus(spt)

        # Smoothing
        if out_avg is None:  # all'iterazione 0 non c'è
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
            # nuovo valore = valore corrente * peso + valore new * (1 - peso)

        # Loss(es)
        # out_loss = mse(out, img_noisy_torch)  # loss 'originaria'

        if square == 1:
            # speckle --> sqrt of Nrif
            im_prod = out * speckle_tensor
        else:
            im_prod = out * torch.sqrt(speckle_tensor)

        spatial_loss = mse(im_prod, img_noisy_torch)  # mse(im_prod, img_noisy_torch)

        h = compute_h_broadcast(speckle_tensor, resh_expanded, step)
        h_loss = histogram_loss(h, h_ref)  # mse istogrammi

        # block_losses = block_hist_loss(speckle_tensor, bh_resh_expanded, step, size=block_size, h_ref=h_ref)
        if args.des_im is not None:
            bo_loss = mse(out, des_im[None, :])
            total_loss = (s_weight * spatial_loss) + (h_weight * h_loss) + bo_loss
        elif loss_fusion is not None:
            bo_loss = mse(out, des_im_fus[None, :])
            total_loss = (s_weight * spatial_loss) + (h_weight * h_loss) + bo_loss
        else:
            total_loss = (s_weight * spatial_loss) + (h_weight * h_loss)

        if int(args.comet) is not None:
            experiment.log_metric('Spatial_loss', spatial_loss.item(), i)
            experiment.log_metric('Histogram_loss', h_loss.item(), i)
            experiment.log_metric('Total_loss', total_loss.item(), i)

            if bo_loss is not None:
                experiment.log_metric('Des_loss', bo_loss.item(), i)

            experiment.log_histogram_3d(speckle_tensor.detach().cpu().numpy(), 'speckle_tensor', i)

            speckle_variance, speckle_mean = torch.var_mean(speckle_tensor, dim=(0, 1, 2), unbiased=False)
            experiment.log_metric('speckle_mean', speckle_mean.detach().cpu(), i)
            experiment.log_metric('speckle_variance', speckle_variance.detach().cpu(), i)

        h_loss_history.append(h_loss.item())
        s_loss_history.append(spatial_loss.item())

        if bo_loss is not None:
            des_loss_history.append(bo_loss.item())

        # Backward
        total_loss.backward()
        loss_history.append(total_loss.item())
        # print("backward done")

        if int(args.gc) == 1:
            # par = get_params(OPT_OVER, net, net_input, spt, downsampler=None)  # devo ricalcolarli ogni volta ?
            torch.nn.utils.clip_grad_value_(p, clip_value=1.0)
            # nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)

        out_depad = out.detach().cpu().numpy()[0]  # (C, H, W)
        out_avg_depad = out_avg.detach().cpu().numpy()[0]
        im_prod_depad = im_prod.detach().cpu().numpy()[0]
        speckle_tensor_detached = speckle_tensor.detach().cpu().numpy()

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
                              'im_prod': im_prod_depad, 'speckle': speckle_tensor_detached,
                              'it': str(earlystop.best_epoch)}

        #####

        print('Iteration %05d   Loss %f ' % (i, total_loss.item()), '\r', end='')

        # if i == 0:
        #     vi_ref = {'out': out_depad, 'out_avg': out_avg_depad, 'im_prod': im_prod_depad,
        #               'speckle': speckle_tensor_detached,
        #               'it': str(i)}
        #     vi_ref_sm = ssim_ref = vi_ref  # same initialization for all  # ssim_ref_sm

        if i == (num_iter - 1):
            # salvo le immagini dell'ultima iterazione
            vi_ref = {'out': out_depad, 'out_avg': out_avg_depad, 'im_prod': im_prod_depad,
                      'speckle': speckle_tensor_detached,
                      'it': str(i)}

        if i == 0 or i % plot_step == 0:
            # todo: qui funziona se tutte le immagini hanno le stesse dimensioni ... se faccio padding effettivamente,
            #  servirà depaddare anche lo speckle ...

            # # Save image
            img_saved = np_to_visual(out_depad)  # (H, W)
            # np.clip(speckle_tensor_detached, 0, 1)
            speckle_saved = np_to_visual(speckle_tensor_detached)  # (H, W)
            plt.imsave(save_path + '/it' + str(i) + '.png', img_saved, cmap='gray')
            plt.imsave(save_path + '/speckle_it' + str(i) + '.png', speckle_saved, cmap='gray')
            # imsave salva le immagini tra min e max dell'immagine, di default se metto cmap='gray'

            # salvo anche le altezze dei bin
            h_data.append({'epoch': str(i), 'h': (h.detach().cpu().numpy()).tolist()})

            # backtracking (?)

        # Update weights
        optimizer.step()

        # sched step
        # sched.step(h_loss)
        # sched.step()

        i += 1

    end = time.time()
    print("Execution time: ", end - start)

    experiment.end()

    log_data.append({'execution_time': end - start})
    log_data.append({'experiment_key': ek})

    f = open(save_path + '/loss_list.pckl', 'wb')
    pickle.dump(loss_history, f)
    f.close()

    f = open(save_path + '/h_loss_list.pckl', 'wb')
    pickle.dump(h_loss_history, f)
    f.close()

    f = open(save_path + '/s_loss_list.pckl', 'wb')
    pickle.dump(s_loss_history, f)
    f.close()

    if bo_loss is not None:
        f = open(save_path + '/des_loss_list.pckl', 'wb')
        pickle.dump(des_loss_history, f)
        f.close()

    f = open(save_path + '/variance_history.pckl', 'wb')
    pickle.dump([x_axis, variance_history], f)
    f.close()

    img_saved = np_to_visual(vi_ref['out'])
    img_saved_sm = np_to_visual(vi_ref['out_avg'])
    img_prod_saved = np_to_visual(vi_ref['im_prod'])
    speckle_saved = np_to_visual(vi_ref['speckle'])

    # mean, variance
    speckle_mean = np.mean(speckle_saved)
    speckle_variance = np.var(speckle_saved)

    # Save as mat and pickle
    scipy.io.savemat(save_path + '/images_it' + str(num_iter - 1) + '.mat',
                     mdict={'img_saved': img_saved, 'img_saved_sm': img_saved_sm,
                            'img_prod_saved': img_prod_saved,
                            'speckle_saved': speckle_saved})

    # f = open(save_path + '/images_it' + str(num_iter - 1) + '.pckl', 'wb')
    # pickle.dump([img_saved, img_saved_sm, img_prod_saved, speckle_saved], f)
    # f.close()

    # Save as images
    plt.imsave(save_path + '/img_it' + vi_ref['it'] + '.png', img_saved, cmap='gray')
    plt.imsave(save_path + '/img_sm_it' + vi_ref['it'] + '.png', img_saved_sm, cmap='gray')
    plt.imsave(save_path + '/img_prod_it' + vi_ref['it'] + '.png', img_prod_saved, cmap='gray')
    plt.imsave(save_path + '/speckle_it' + vi_ref['it'] + '.png', speckle_saved, cmap='gray')

    if best_updated:

        # Save ES-WMV images
        img_es_saved = np_to_visual(es_ref['out'])
        img_es_sm_saved = np_to_visual(es_ref['out_avg'])
        img_es_prod_saved = np_to_visual(es_ref['im_prod'])
        speckle_es_saved = np_to_visual(es_ref['speckle'])

        # mean, variance
        speckle_es_mean = np.mean(speckle_es_saved)
        speckle_es_variance = np.var(speckle_es_saved)

        scipy.io.savemat(save_path + '/images_max_es.mat', mdict={'img_es_saved': img_es_saved,
                                                                    'img_es_sm_saved': img_es_sm_saved,
                                                                    'img_es_prod_saved': img_es_prod_saved,
                                                                    'speckle_es_saved': speckle_es_saved})

        plt.imsave(save_path + '/img_es_max_it' + es_ref['it'] + '.png', img_es_saved, cmap='gray')
        plt.imsave(save_path + '/img_es_max_sm_it' + es_ref['it'] + '.png', img_es_sm_saved, cmap='gray')
        plt.imsave(save_path + '/img_es_max_prod_it' + es_ref['it'] + '.png', img_es_prod_saved, cmap='gray')
        plt.imsave(save_path + '/speckle_es_max_it' + es_ref['it'] + '.png', speckle_es_saved, cmap='gray')

        log_data.append({'best_epoch_ES': es_ref['it'],
                        'speckle_mean': speckle_es_mean.astype('float64'),
                        'speckle_variance': speckle_es_variance.astype('float64')})
    
    # log
    log_data.append({'last_epoch': vi_ref['it'],
                    'speckle_mean': speckle_mean.astype('float64'),
                    'speckle_variance': speckle_variance.astype('float64')})

    write_json(log_data, save_path + '/log_file.json')

    # Save the bin heights
    write_json(h_data, save_path + '/h_file.json')
