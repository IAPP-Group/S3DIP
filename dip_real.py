""" DIP for real SAR images """

from __future__ import print_function
from comet_ml import Experiment, OfflineExperiment
from models import get_net
from utils_mat.common_utils_mat import get_params, get_noise, np_to_torch, torch_to_np, np_to_visual
import argparse
import pickle
import os
import numpy as np
import torch
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from functions import write_json
import time
from stopping_criterion import EarlyStop, myMetric

# nb: mi serve float32 per il psnr mi sa ...
# nb: mi serve float64 per il laplaciano

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DIP applied to real SAR images")
    parser.add_argument("--pkl", dest="input", default=None, help="path to pickle file") 
    parser.add_argument("--base_save_path", dest="base_save_path", default=None,
                        help="path to the base folder for results. here we save the new folder for current experiment")
    parser.add_argument("--base_dataset_path", dest="base_dataset_path", default=None,
                        help="path to the base dataset folder. here we have pckl_files, SAR_BM3D and FANS folders")
    parser.add_argument("--iter", dest="num_iter", default=5, help="Number of iteration")
    parser.add_argument("--num_channels_out", dest="num_channels_out", default=1,  # 3
                        help="Number of channels output image")
    parser.add_argument("--input_depth", dest="input_depth", default=8,
                        help="3/1 for gt == 0, 32 for gt == 1")
    parser.add_argument("--plot", dest="plot", default=None,  # None
                        help="True if we want to plot current output of the net. None otherwise")
    parser.add_argument("--plot_step", dest="plot_step", default=10, help="Step for plots/save images during training")

    parser.add_argument("--lr", dest="lr", default=0.01, help="Learning rate")

    # comet parameters
    parser.add_argument("--comet", dest="comet", default=0, help="1 for comet ON, 0 for comet OFF")
    parser.add_argument("--name_proj", dest="name_proj", default='despeckling_paper', help="define comet ml project folder")
    parser.add_argument("--name_exp", dest="name_exp", default='we2', help="name of comet ml experiment")
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
    torch.backends.cudnn.benchmark = True  # cuDNN Autotuner on
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
    PLOT = bool(args.plot)
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
    f = open(args.input, 'rb')
    noisy_image = pickle.load(f)

    # noisy image
    # img_noisy, h_pad_noisy, w_pad_noisy = padding(noisy_image, d=input_depth)  # img_noisy_pil
    img_noisy = noisy_image[None, :]
    img_noisy = img_noisy.astype(np.float32)  # nd.array (C, H, W)
    img_noisy_torch = np_to_torch(img_noisy).to(device)  # type(dtype)
    print("img noisy torch shape: ", img_noisy_torch.shape)

    if args.des_im is not None:
        f = open(args.des_im, 'rb')
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

        n_looks = args.input.split('/')[-3]  # L1 in .../L1/pckl_files/nome.pckl

        # fname_s = 'D:/DATASET_SAR/SAR-BM3D/pckl_files/' + loss_fusion[:-5] + '/' + loss_fusion + '_sar-bm3d.pckl'
        fname_s = args.base_dataset_path + '/' + n_looks + '/SAR-BM3D/pckl_files/' + loss_fusion + '_sar-bm3d.pckl'
        f = open(fname_s.rstrip('\r'), 'rb')
        des_im_s = pickle.load(f)  # nd.array nel range (0,1)
        des_im_s = np_to_torch(des_im_s.astype(np.float32))
        f.close()

        # fname_f = 'D:/DATASET_SAR/FANS/pckl_files/' + loss_fusion[:-5] + '/' + loss_fusion + '_fans.pckl'  # local
        fname_f = args.base_dataset_path + '/' + n_looks + '/FANS/pckl_files/' + loss_fusion + '_fans.pckl'
        # .../REAL_IMAGES/L1/FANS/pckl_files/...

        f = open(fname_f.rstrip('\r'), 'rb')
        des_im_f = pickle.load(f)  # nd.array nel range (0,1)
        des_im_f = np_to_torch(des_im_f.astype(np.float32))
        f.close()

        des_im_fus = ((des_im_s + des_im_f) / 2).to(device)
        print("des im fus shape: ", des_im_fus.shape)

    # Z net input (not under the computational graph)
    # net_input = get_noise(input_depth, INPUT, (img.shape[1], img.shape[2])).to(device).detach()  # img è C, H, W
    net_input = get_noise(input_depth, INPUT, (img_noisy.shape[1], img_noisy.shape[2])).to(device).detach()
    print("net input shape: ", net_input.shape)

    # NET
    net = get_net(input_depth, 'skip', pad, n_channels=nco,  # ho un'immagine in bianco e nero
                  skip_n33d=int(args.skip_n33d),
                  skip_n33u=int(args.skip_n33u),
                  skip_n11=int(args.skip_n11),
                  num_scales=int(args.num_scales),
                  upsample_mode='bilinear',
                  act_fun=str(args.act_fun)).to(device)  # type(dtype)

    # Compute number of parameters
    s = sum([np.prod(list(p.size())) for p in net.parameters()])
    print('Number of NET params: %d' % s)

    # Optimization
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None

    p = get_params(OPT_OVER, net, net_input, noise_tensor=None, downsampler=None)  # total params to optimize
    s = sum([np.prod(list(g.size())) for g in p])
    print('Number of TOTAL params: %d' % s)

    optimizer = torch.optim.Adam(p, lr=LR)  # weight_decay=

    # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=40, threshold=0.0001,
    #                                                    threshold_mode='rel',
    #                                                    cooldown=0, min_lr=0, eps=1e-08, verbose=False)

    # Loss(es)
    mse = torch.nn.MSELoss().to(device)  # type(dtype)

    # COMET
    experiment = None
    if int(args.comet) == 0:
        # Comet ml integration
        experiment = OfflineExperiment(offline_directory=
                                       '/Prove/Albisani/DIP_DESPECKLING/DESPECKLING_RESULTS/COMET_OFFLINE',
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
    log_data = []

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

        # Smoothing
        if out_avg is None:  # all'iterazione 0 non c'è
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
            # nuovo valore = valore corrente * peso + valore new * (1 - peso)

        # Loss
        spatial_loss = mse(out, img_noisy_torch)
        if args.des_im is not None:
            bo_loss = mse(out, des_im[None, :])
            total_loss = spatial_loss + bo_loss

        elif loss_fusion is not None:
            bo_loss = mse(out, des_im_fus[None, :])
            total_loss = spatial_loss + bo_loss

        else:
            total_loss = spatial_loss

        # calcolo lo speckle come rapporto noisy / clean
        # speckle_tensor = img_noisy_torch / out

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

        out_depad = out.detach().cpu().numpy()[0]  # (C, H, W)
        out_avg_depad = out_avg.detach().cpu().numpy()[0]

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
                    es_ref = {'out': out_depad, 'out_avg': out_avg_depad, 'it': str(earlystop.best_epoch)}

        #####

        print('Iteration %05d    Loss %f ' % (i, total_loss.item()), '\r', end='')

        if i == (num_iter - 1):
            # salvo le immagini dell'ultima iterazione
            vi_ref = {'out': out_depad, 'out_avg': out_avg_depad, 'it': str(i)}

        if i == 0 or i % plot_step == 0:
            # # Save image
            img_saved = np_to_visual(out_depad)  # (H, W)
            # np.clip(speckle_tensor_detached, 0, 1)
            plt.imsave(save_path + '/it' + str(i) + '.png', img_saved, cmap='gray')
            # imsave salva le immagini tra min e max dell'immagine, di default se metto cmap='gray'

        # Update weights
        optimizer.step()

        # sched.step(total_loss)

        i += 1

    end = time.time()
    print("Execution time: ", end - start)

    experiment.end()

    log_data.append({'execution_time': end - start})
    log_data.append({'experiment_key': ek})

    f = open(save_path + '/loss_list.pckl', 'wb')
    pickle.dump(loss_history, f)
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

    scipy.io.savemat(save_path + '/images_it' + str(num_iter - 1) + '.mat',
                     mdict={'img_saved': img_saved, 'img_saved_sm': img_saved_sm})
    # f = open(save_path + '/images_it' + str(num_iter - 1) + '.pckl', 'wb')
    # pickle.dump([img_saved, img_saved_sm], f)
    # f.close()

    # Save as images
    plt.imsave(save_path + '/img_it' + vi_ref['it'] + '.png', img_saved, cmap='gray')
    plt.imsave(save_path + '/img_sm_it' + vi_ref['it'] + '.png', img_saved_sm, cmap='gray')

    if best_updated:

        # Save ES-WMV images
        img_es_saved = np_to_visual(es_ref['out'])
        img_es_sm_saved = np_to_visual(es_ref['out_avg'])

        scipy.io.savemat(save_path + '/images_max_es.mat', mdict={'img_es_saved': img_es_saved,
                                                                'img_es_sm_saved': img_es_sm_saved})

        plt.imsave(save_path + '/img_es_max_it' + es_ref['it'] + '.png', img_es_saved, cmap='gray')
        plt.imsave(save_path + '/img_es_max_sm_it' + es_ref['it'] + '.png', img_es_sm_saved, cmap='gray')

        # log
        # log_data.append({'last_epoch': vi_ref['it']})

        log_data.append({'best_epoch_ES': es_ref['it']})

    write_json(log_data, save_path + '/log_file.json')
